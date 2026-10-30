import pickle
import numpy as np
import torch
from test_b1k import root,make_dataset
from diffusion_policy.b1k.b1k_cache_v2 import build_group
from diffusion_policy.b1k.frame_cache import selected_videos,verify_frame_cache
from diffusion_policy.b1k.robot import CAMERAS
from diffusion_policy.b1k.model import ModelConfig


def test_all_frame_anchors_padding_stats_and_absolute_grippers(root):
    d=make_dataset(root,observation_mode='lowdim',frame_selection='pi',gripper_state='sum',
                   action_representation='relative',gripper_action='absolute',stats_estimator='exact',return_meta=True,task_onehot=False)
    try:
        assert len(d)==13
        stats=d.compute_action_stats('quantile');values=[];anchors=[]
        for i in range(len(d)):
            sample=d[i];values.extend(sample['action'].numpy()[sample['action_valid'].numpy()])
            anchors.append(int(sample['meta']['frame']))
        assert anchors==list(range(5))+list(range(8))
        values=np.asarray(values,dtype=np.float64)
        assert stats['action_count']==len(values)==44
        np.testing.assert_allclose(stats['action']['q01'],np.quantile(values,.01,axis=0),atol=1e-6)
        np.testing.assert_allclose(stats['action']['std'],values.std(0),atol=1e-6)
        assert stats['action_delta']['gripper_action_dims']==[]
        assert ModelConfig().frame_selection=='legacy' and ModelConfig().gripper_action=='delta'
    finally:d.close()


def test_lazy_goals_mmaps_v2_and_worker_pickle(root,tmp_path):
    native=make_dataset(root,resize_mode='act',goal_views=['head'])
    cache=tmp_path/'cache'
    for path in selected_videos(native):build_group([path],cache,root,(16,16))
    assert verify_frame_cache(native,cache,samples=4,log=lambda *_:None)>0
    cached=make_dataset(root,resize_mode='act',frame_cache=cache,goal_views=['head'],image_dtype='uint8')
    try:
        assert isinstance(cached._sidecar['state'],np.memmap)
        assert not isinstance(cached.goal_table,np.ndarray)
        assert len(pickle.dumps(cached))<100_000
        restored=pickle.loads(pickle.dumps(cached))
        for i in range(len(cached)):
            first,second=cached[i],restored[i]
            torch.testing.assert_close(first['action'],second['action'])
            for k in first['obs']:torch.testing.assert_close(first['obs'][k],second['obs'][k])
        restored.close()
        loader=torch.utils.data.DataLoader(cached,batch_size=3,num_workers=1,multiprocessing_context='spawn')
        assert sum(len(batch['action']) for batch in loader)==len(cached)
    finally:native.close();cached.close()


def test_v2_training_resume_and_quantile_sampler(root,tmp_path):
    from diffusion_policy.b1k.train import main
    native=make_dataset(root,cameras=['head'],resize_mode='act')
    cache=tmp_path/'cache'
    for path in selected_videos(native):build_group([path],cache,root,(16,16))
    native.close()
    common=['--dataset-root',str(root),'--device','cpu','--num-workers','0','--batch-size','2',
            '--cpu-threads','1','--horizon','4','--n-action-steps','3','--image-size','16','--cameras','head',
            '--goal-views','head','--goal-fusion','late','--variant','unet_hybrid_image','--resize-mode','act','--frame-cache',str(cache),
            '--down-dims','16','32','--diffusion-step-embed-dim','16','--num-train-timesteps','4',
            '--num-inference-steps','2','--action-representation','relative','--normalization','quantile']
    full=tmp_path/'full';split=tmp_path/'split'
    main(common+['--output-dir',str(full),'--max-steps','3'])
    main(common+['--output-dir',str(split),'--max-steps','2'])
    main(common+['--output-dir',str(split),'--max-steps','3','--resume',str(split/'latest.pt')])
    a=torch.load(full/'latest.pt',weights_only=True);b=torch.load(split/'latest.pt',weights_only=True)
    for field in ('model','ema_model'):
        for key in a[field]:torch.testing.assert_close(a[field][key],b[field][key],rtol=0,atol=0)
    assert b['config']['gripper_action']=='absolute' and b['config']['frame_selection']=='pi'
    assert b['config']['clip_sample'] is False


def test_lowdim_precomputation_loads_in_rgb_training(root,tmp_path):
    import shutil
    from diffusion_policy.b1k.stats_v2 import main,load_or_compute
    copy=tmp_path/'lowdim';shutil.copytree(root,copy,ignore=shutil.ignore_patterns('videos'))
    stats=tmp_path/'stats.json'
    main(['--dataset-root',str(copy),'--output',str(stats),'--episode-split','none','--settle-steps','all',
          '--val-fraction','0','--horizon','4','--n-obs-steps','2','--n-action-steps','3'])
    rgb=make_dataset(root,gripper_state='sum',task_onehot=False,
        action_representation='relative',frame_selection='pi',gripper_action='absolute',stats_estimator='exact')
    try:
        values=load_or_compute(rgb,'quantile',stats,tmp_path)
        expected=rgb.compute_action_stats('quantile')
        np.testing.assert_array_equal(values['action']['q99'],expected['action']['q99'])
    finally:rgb.close()
