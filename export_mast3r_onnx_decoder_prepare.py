#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import torch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.model_encoder import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser

torch.backends.cuda.matmul.allow_tf32 = True
batch_size = 1

def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--input_images', nargs='+', required=True, help='Input image files')
    parser.add_argument('--optim_level', choices=['coarse', 'refine', 'refine+depth'], default='refine', help='Optimization level')
    parser.add_argument('--lr1', type=float, default=0.07, help='Coarse learning rate')
    parser.add_argument('--niter1', type=int, default=500, help='Number of coarse iterations')
    parser.add_argument('--lr2', type=float, default=0.014, help='Fine learning rate')
    parser.add_argument('--niter2', type=int, default=200, help='Number of fine iterations')
    parser.add_argument('--scenegraph_type', choices=['complete', 'swin', 'logwin', 'oneref'], default='complete', help='Scene graph type')
    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    parser.prog = 'mast3r demo'
    return parser

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i].reshape(imgs[i].shape), mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)


def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, optim_level, lr1, niter1, lr2, niter2,
                            min_conf_thr, matching_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams,
                            cam_size, scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics,
                            **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    scene = sparse_global_alignment(filelist, pairs, os.path.join(outdir, 'cache'),
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)
    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)
    return scene, outfile


def main(args):
    model = AsymmetricMASt3R.from_pretrained(args.weights or "naver/" + args.model_name).to(args.device)
    chkpt_tag = hash_md5(args.weights or "naver/" + args.model_name)

    output_dir = os.path.join(args.output_dir, chkpt_tag)
    os.makedirs(output_dir, exist_ok=True)

    scene, outfile = get_reconstructed_scene(
        output_dir, model, args.device, args.silent, args.image_size,
        args.input_images, 'refine', 0.07, 500, 0.014, 200, 1.5, 5.0,
        True, False, True, False, 0.2, 'complete', 1, False, 0, 0.0, False
    )

    print(f"3D model saved to: {outfile}")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    imgs = load_images(args.input_images, size=args.image_size, verbose=not args.silent)
    torch_model = AsymmetricMASt3R.from_pretrained(args.weights or "naver/" + args.model_name).to(args.device)
    print(torch_model)
    #"AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
    chkpt_tag = hash_md5(args.weights or "naver/" + args.model_name)

    output_dir = os.path.join(args.output_dir, chkpt_tag)
    os.makedirs(output_dir, exist_ok=True)
    img1, img2 = imgs[0], imgs[1]
    shape1 = torch.from_numpy(img1['true_shape']).to(args.device, non_blocking=True)
    shape2 = torch.from_numpy(img2['true_shape']).to(args.device, non_blocking=True)
    img1 = img1['img'].to(args.device, non_blocking=True)
    img2 = img2['img'].to(args.device, non_blocking=True)
    print(img1.shape, img2.shape)
    input = (img1, img2, shape1, shape2)
    feat1, feat2, pos1, pos2 = torch_model(img1, img2, shape1, shape2)
    torch.save(feat1, 'input'+'/feat1.pth')
    torch.save(feat2, 'input'+'/feat2.pth')
    torch.save(pos1, 'input'+'/pos1.pth')
    torch.save(pos2, 'input'+'/pos2.pth')
    torch.save(shape1, 'input'+'/shape1.pth')
    torch.save(shape2, 'input'+'/shape2.pth')
    
    # torch.onnx.export(torch_model, input, os.path.join(output_dir, 'mast3r_encoder_params.onnx'), export_params=True, opset_version=17, do_constant_folding=True, verbose=True)
