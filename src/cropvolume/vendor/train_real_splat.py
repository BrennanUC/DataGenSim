"""Fit real-photo 3DGS from recovered cameras with withheld render views and inspectable outputs."""
# Shared copy of tools/train_real_splat.py from the September 2026 basil splat run on RunPod. Edits for sharing: the work folder comes from $SPLAT_WORK and the RunPod-only checks are removed; method and settings are unchanged.
from pathlib import Path
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import time
import traceback

import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
import torch
from torch import nn
from gsplat import DefaultStrategy,rasterization
from gsplat.strategy.ops import reset_opa
import gsplat
from pytorch_msssim import ssim
from plyfile import PlyData,PlyElement

ROOT=Path(os.environ.get('SPLAT_WORK') or Path(__file__).resolve().parents[1])
C0=.28209479177387814

def write(path,data):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(path)

def status(out,state,**extra):
    row=dict(state=state,pid=os.getpid(),epoch=time.time(),**extra)
    write(out/'status.json',row);print(json.dumps(row),flush=True)

def render(params,view,K,width,height,degree=3,absgrad=False,antialiased=False,depth=False):
    return rasterization(means=params['means'],quats=params['quats'],scales=params['scales'].exp(),
        opacities=params['opacities'].sigmoid(),colors=torch.cat([params['sh0'],params['shN']],1),sh_degree=degree,
        viewmats=view,Ks=K,width=width,height=height,near_plane=.05,far_plane=2000,
        backgrounds=torch.zeros((1,3),device='cuda'),packed=False,absgrad=absgrad,
        rasterize_mode='antialiased' if antialiased else 'classic',render_mode='RGB+ED' if depth else 'RGB')

def export_ply(path,params):
    arrays={k:v.detach().cpu().numpy() for k,v in params.items()};n=len(arrays['means'])
    quats=arrays['quats']/np.maximum(np.linalg.norm(arrays['quats'],axis=1,keepdims=True),1e-12)
    rest=arrays['shN'].transpose(0,2,1).reshape(n,-1)
    names=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(rest.shape[1])]
    names+=['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    values=np.concatenate([arrays['means'],np.zeros((n,3),np.float32),arrays['sh0'].reshape(n,3),rest,arrays['opacities'][:,None],arrays['scales'],quats],1).astype('<f4')
    assert np.isfinite(values).all() and len(names)==values.shape[1]
    rows=np.empty(n,dtype=[(name,'<f4') for name in names])
    for i,name in enumerate(names):rows[name]=values[:,i]
    PlyData([PlyElement.describe(rows,'vertex')],text=False,byte_order='<').write(str(path))
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(gaussians=n,bytes=path.stat().st_size,sha256=digest,sh_degree=3,world_coordinates='Approximate DJI-GPS-aligned ENU metres, Z up; not ground-control validated',quaternions='wxyz',scales='log standard deviation',opacities='logit')

def camera(frame,width):
    width=min(width,frame['width']);height=round(frame['height']*width/frame['width'])
    K=np.array(frame['K'],np.float32);K[0]*=width/frame['width'];K[1]*=height/frame['height']
    view=np.linalg.inv(np.array(frame['c2w_cv'],np.float32))
    return width,height,K,view

class Images:
    def __init__(self,root,frames,width,cache_limit_gb=24):
        self.root,self.frames,self.width=root,frames,width;self.cache={};self.bytes=0;self.limit=cache_limit_gb*1024**3
    def get(self,index,width=None):
        width=width or self.width;key=(index,width)
        if key not in self.cache:
            frame=self.frames[index];w,h,_,_=camera(frame,width)
            with Image.open(self.root/frame['image']) as image:array=np.asarray(image.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)).copy()
            if self.bytes+array.nbytes<self.limit:self.cache[key]=array;self.bytes+=array.nbytes
            return array
        return self.cache[key]

@torch.no_grad()
def evaluate(params,data,images,indices,out,step,width,antialiased=False,label='validation'):
    folder=out/(label+f'-{step:06d}');folder.mkdir(exist_ok=True);rows=[]
    for j,index in enumerate(indices):
        frame=data['frames'][index];w,h,K,view=camera(frame,width)
        rendered,alpha,_=render(params,torch.tensor(view,device='cuda')[None],torch.tensor(K,device='cuda')[None],w,h,antialiased=antialiased,depth=True)
        truth=torch.tensor(images.get(index,width),device='cuda',dtype=torch.float32)[None]/255
        color=rendered[...,:3].clamp(0,1);mse=float((color-truth).square().mean());psnr=-10*math.log10(max(mse,1e-12))
        score=float(ssim(color.permute(0,3,1,2),truth.permute(0,3,1,2),data_range=1,size_average=True))
        green=(truth[...,1]>truth[...,0]*1.06)&(truth[...,1]>truth[...,2]*1.12)
        gmse=float((color-truth).square()[green].mean()) if int(green.sum())>0 else None
        rows.append(dict(name=frame['name'],split=frame['split'],psnr_db=psnr,ssim=score,
            green_psnr_db=-10*math.log10(max(gmse,1e-12)) if gmse is not None else None,width=w,height=h,
            alpha_coverage_fraction=float((alpha>.5).float().mean())))
        if j<8:
            reference=images.get(index,width);predicted=(color[0].cpu().numpy()*255+.5).astype(np.uint8)
            stem=f'{index:05d}';Image.fromarray(predicted).save(folder/(stem+'-splat.png'))
            Image.fromarray(reference).save(folder/(stem+'-reference.jpg'),quality=98)
            display_width=min(w,1200);display_height=round(h*display_width/w)
            pair=Image.new('RGB',(display_width*2,display_height+42),(22,29,25));draw=ImageDraw.Draw(pair)
            draw.text((16,14),'Withheld photograph',fill='white');draw.text((display_width+16,14),f'Learned Gaussian render | {psnr:.2f} dB',fill='white')
            pair.paste(Image.fromarray(reference).resize((display_width,display_height)),(0,42))
            pair.paste(Image.fromarray(predicted).resize((display_width,display_height)),(display_width,42));pair.save(folder/(stem+'-comparison.jpg'),quality=95)
            np.savez_compressed(folder/(stem+'-depth-alpha.npz'),depth=rendered[0,:,:,3].cpu().numpy().astype(np.float16),alpha=alpha[0,:,:,0].cpu().numpy().astype(np.float16))
    assert rows
    result=dict(step=step,views=rows,mean_psnr_db=float(np.mean([x['psnr_db'] for x in rows])),mean_ssim=float(np.mean([x['ssim'] for x in rows])),holdout_scope=data['holdout_scope'])
    write(folder/'metrics.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--run',required=True)
    p.add_argument('--steps',type=int,default=12000);p.add_argument('--width',type=int,default=1320);p.add_argument('--eval-width',type=int,default=2640)
    p.add_argument('--init-points',type=int,default=500000);p.add_argument('--max-gaussians',type=int,default=2000000)
    p.add_argument('--absgrad',action='store_true');p.add_argument('--antialiased',action='store_true')
    p.add_argument('--grow-grad',type=float);p.add_argument('--grow-scale',type=float,default=.002)
    p.add_argument('--seed',type=int,default=923);p.add_argument('--all-views',action='store_true');p.add_argument('--patch-size',type=int,default=0)
    p.add_argument('--share-gpu',action='store_true',help='Permit a bounded pilot alongside feature extraction; elapsed time is not an isolated performance benchmark.')
    a=p.parse_args();out=ROOT/'results/models'/a.run;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'checkpoint.pt').exists() and not (out/'metrics.json').exists(),'Do not overwrite trained evidence'
    dataset=ROOT/'data/prepared'/a.case;data=json.loads((dataset/'dataset.json').read_text());assert data['schema']=='cropsplat-real-colmap-v1'
    write(out/'config.json',dict(options=vars(a),dataset_sha256=hashlib.sha256((dataset/'dataset.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (out/'trainer.py').write_bytes(Path(__file__).read_bytes())
    try:
        gpu_lock=(ROOT/'runs/gpu-work.lock').open('a');status(out,'waiting_for_gpu')
        if not a.share_gpu:fcntl.flock(gpu_lock,fcntl.LOCK_EX)
        else:
            training_lock=(ROOT/'runs/splat-training.lock').open('a')
            fcntl.flock(training_lock,fcntl.LOCK_EX)
        torch.manual_seed(a.seed);torch.set_num_threads(8);rng=np.random.default_rng(a.seed)
        assert torch.cuda.is_available()
        train=np.array([i for i,f in enumerate(data['frames']) if a.all_views or f['split']=='train'])
        valid=[i for i,f in enumerate(data['frames']) if f['split']=='validation']
        test=[i for i,f in enumerate(data['frames']) if f['split']=='test']
        assert len(train)>20 and valid and test
        saved=np.load(dataset/data['initialization']);points=saved['points'];colors=saved['colors'].astype(np.float32)/255
        if len(points)>a.init_points:
            chosen=rng.choice(len(points),a.init_points,replace=False);points,colors=points[chosen],colors[chosen]
        distances=cKDTree(points).query(points,k=4,workers=8)[0][:,1:]
        scales=np.sqrt((distances**2).mean(1)).clip(.002,.3)
        n=len(points);sh=np.zeros((n,16,3),np.float32);sh[:,0]=(colors-.5)/C0
        quats=rng.normal(size=(n,4)).astype(np.float32);quats/=np.linalg.norm(quats,axis=1,keepdims=True)
        params=nn.ParameterDict({
            'means':nn.Parameter(torch.tensor(points,device='cuda')),
            'scales':nn.Parameter(torch.tensor(np.log(np.repeat(scales[:,None],3,axis=1)),device='cuda',dtype=torch.float32)),
            'quats':nn.Parameter(torch.tensor(quats,device='cuda')),
            'opacities':nn.Parameter(torch.full((n,),float(torch.logit(torch.tensor(.1))),device='cuda')),
            'sh0':nn.Parameter(torch.tensor(sh[:,:1],device='cuda')),'shN':nn.Parameter(torch.tensor(sh[:,1:],device='cuda'))})
        local_scale=data['camera_clearance_m'];rates={'means':1.6e-4*local_scale,'scales':.005,'quats':.001,'opacities':.05,'sh0':.0025,'shN':.0025/20}
        opts={k:torch.optim.Adam([params[k]],lr=lr,eps=1e-15) for k,lr in rates.items()}
        grow_grad=a.grow_grad if a.grow_grad is not None else .0008 if a.absgrad else .0002
        stop_refine=round(a.steps*.7);reset_every=6000
        strategy=DefaultStrategy(refine_start_iter=500,refine_stop_iter=stop_refine,refine_every=100,reset_every=reset_every,
            pause_refine_after_reset=min(len(train),1000),grow_grad2d=grow_grad,grow_scale3d=a.grow_scale,
            prune_scale3d=.1,refine_scale2d_stop_iter=0,absgrad=a.absgrad,verbose=False)
        strategy.check_sanity(params,opts);state=strategy.initialize_state(scene_scale=local_scale)
        images=Images(dataset,data['frames'],a.width,cache_limit_gb=30)
        views=torch.tensor(np.linalg.inv(np.array([f['c2w_cv'] for f in data['frames']],np.float32)),device='cuda')
        camera_cache={i:camera(data['frames'][i],a.width)[:3] for i in train}
        del points,colors,scales,quats,sh,distances;gc.collect()
        seen=np.zeros(len(data['frames']),np.int64);history=[];started=time.monotonic();capped=None
        order=rng.permutation(train);cursor=0;torch.cuda.reset_peak_memory_stats()
        for step in range(1,a.steps+1):
            if cursor==len(order):order=rng.permutation(train);cursor=0
            i=int(order[cursor]);cursor+=1;seen[i]+=1
            w,h,K=camera_cache[i];K=K.copy();source=images.get(i)
            if a.patch_size and min(w,h)>a.patch_size and step>1000 and step%5:
                x=int(rng.integers(w-a.patch_size+1));y=int(rng.integers(h-a.patch_size+1))
                source=source[y:y+a.patch_size,x:x+a.patch_size];K[0,2]-=x;K[1,2]-=y;w=h=a.patch_size
            target=torch.tensor(np.ascontiguousarray(source),device='cuda',dtype=torch.float32)[None]/255
            degree=min(3,step//1000)
            color,_,info=render(params,views[i:i+1],torch.tensor(K,device='cuda')[None],w,h,degree=degree,absgrad=a.absgrad,antialiased=a.antialiased)
            loss=.8*(color-target).abs().mean()+.2*(1-ssim(color.permute(0,3,1,2),target.permute(0,3,1,2),data_range=1,size_average=True))
            strategy.step_pre_backward(params,opts,state,step,info);loss.backward()
            for opt in opts.values():opt.step();opt.zero_grad(set_to_none=True)
            opts['means'].param_groups[0]['lr']=rates['means']*(.01**(step/a.steps))
            strategy.step_post_backward(params,opts,state,step,info,packed=False)
            # Correct the v1.5.3 bitwise reset-condition bug without altering the pinned package.
            if step%reset_every==0 and step<strategy.refine_stop_iter:reset_opa(params,opts,state,value=.01)
            if len(params['means'])>=a.max_gaussians and capped is None:
                capped=step;strategy.grow_grad2d=float('inf')
                # Continue opacity/scale pruning after reaching capacity.
            if step==1 or step%100==0:
                assert torch.isfinite(loss)
                row=dict(step=step,loss=float(loss.detach()),gaussians=len(params['means']),elapsed_s=time.monotonic()-started,gpu_peak_gb=torch.cuda.max_memory_allocated()/1024**3)
                history.append(row);write(out/'history.json',history);status(out,'training',**row)
            if step in {min(3000,a.steps),a.steps}:
                checkpoint=dict(params={k:v.detach().cpu() for k,v in params.items()},step=step,case=a.case,options=vars(a),dataset_sha256=hashlib.sha256((dataset/'dataset.json').read_bytes()).hexdigest())
                temporary=out/'checkpoint.tmp';torch.save(checkpoint,temporary);temporary.replace(out/'checkpoint.pt')
                evaluate(params,data,images,valid if len(valid)<=20 else valid[::math.ceil(len(valid)/20)],out,step,a.eval_width,antialiased=a.antialiased,label='training-view-review' if a.all_views else 'validation')
        assert (seen[train]>0).all();torch.cuda.synchronize();elapsed=time.monotonic()-started
        status(out,'exporting',gaussians=len(params['means']));ply=export_ply(out/'field.ply',params);write(out/'ply-manifest.json',ply)
        final=evaluate(params,data,images,test if len(test)<=32 else test[::math.ceil(len(test)/32)],out,a.steps,a.eval_width,antialiased=a.antialiased,label='training-view-final' if a.all_views else 'test')
        np.save(out/'camera-sample-counts.npy',seen)
        result=dict(state='complete',options=vars(a),initial_gaussians=n,final_gaussians=len(params['means']),cap_reached_step=capped,
            gpu=torch.cuda.get_device_name(0),torch=torch.__version__,gsplat=gsplat.__version__,elapsed_s=elapsed,
            peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,train_views=len(train),test=final,ply=ply,
            evaluation_is_withheld_from_splat_loss=not a.all_views,holdout_scope=data['holdout_scope'],yield_validated=False,
            learning_rate_scene_scale_m=local_scale,scene_scale_choice='Recovered local camera-to-canopy height, not entire field extent.')
        result['concurrent_feature_work_permitted']=a.share_gpu
        write(out/'metrics.json',result);status(out,'complete',gaussians=len(params['means']),test_psnr_db=final['mean_psnr_db'])
    except BaseException as error:
        status(out,'failed',error=str(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
