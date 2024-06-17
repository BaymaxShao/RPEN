import datetime
import torch
import torch.optim
import torch.utils.data
from dataloader import *
from dataloader_opt import *
from loss import *
from utils import *
import numpy as np
import os
import models
from configs import Tester
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    start = time.perf_counter()
    test_obj = [folder.split('/')[0] for folder in open('test_file.txt')][0]
    args = Tester().parse()
    weights = torch.load(args.pretrained_model1)
    model = models.ab6().to(device)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(weights['state_dict'], strict=True)
    model.eval()
    print('=== Pretrained Model Loaded ===')

    test_set = SeqData(
        args.data,
        istrain=False,
        img_size=224,
        train_list='train_file.txt',
        test_list='test_file.txt',
        step=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    ATEs, RTEs, ROTs, CEs, DEs, traj_pred, traj_gt, d_pred, d_gt, rel_pred, rel_gt, length = test(test_loader, model)
    end = time.perf_counter()
    print("inference time：", end - start)
    print("inference speed：", 1/(end - start)*len(test_loader))

    print('The average ATE of the trajectory: {:.4f}'.format(np.mean(ATEs)))
    print('The average RTE of the trajectory: {:.4f}'.format(np.mean(RTEs)))
    print('The average ROT of the trajectory: {:.4f}'.format(np.mean(ROTs)))
    print('The average DE of the trajectory: {:.4f}'.format(np.mean(DEs)))
    print('The average CE of the trajectory: {:.4f}'.format(np.mean(CEs)))

    print('The std ATE of the trajectory: {:.4f}'.format(np.std(ATEs)))
    print('The std RTE of the trajectory: {:.4f}'.format(np.std(RTEs)))
    print('The std ROT of the trajectory: {:.4f}'.format(np.std(ROTs)))
    print('The std DE of the trajectory: {:.4f}'.format(np.std(DEs)))
    print('The std CE of the trajectory: {:.4f}'.format(np.std(CEs)))

    print(length)

    save_results(args.save_path, test_obj, ATEs, RTEs, ROTs, CEs, DEs, traj_pred, traj_gt, d_pred, d_gt, rel_pred, rel_gt)


def test(test_loader, model):
    traj_gt = []
    traj_pred = []
    Ts_gt = []
    Ts_pred = []
    ATEs = []
    RTEs = []
    ROTs = []
    CEs = []
    DEs = []
    d_gt = []
    d_pred = []
    rel_poses_pred = []
    rel_poses_gt = []
    next_pose_pred = None
    length = 0
    for i, (img1, img2, ref_pose, tar_pose, rel_pose) in enumerate(test_loader):
        length += cal_distance(ref_pose[0, 4:], tar_pose[0, 4:])
        tar_pose_T = quat2T(tar_pose)
        if i == 0:
            # Add the start pose
            ref_pose_T = quat2T(ref_pose)
            traj_pred.append(ref_pose_T)
            traj_gt.append(ref_pose_T)
        else:
            ref_pose_T = next_pose_pred
        next_pose_gt = tar_pose_T

        # Load the input
        img1 = img1.to(device)
        img2 = img2.to(device)

        # Predict the relative transformation [t,r]
        rel_pose_pred = model(img1, img2)

        # Transform to [q,t]
        rot_quat_pred = logq_to_quaternion(rel_pose_pred[:, 3:])
        rel_pose_pred = torch.cat([rot_quat_pred, rel_pose_pred[:, :3]], 1).detach().cpu().numpy()

        # Transform to Matrix
        rel_pose_gt = np.array(rel_pose)
        T_pred = quat2T(rel_pose_pred)
        T_gt = quat2T(rel_pose_gt)

        # Predict the next pose
        next_pose_pred = np.matmul(ref_pose_T, T_pred)


        # Update the trajectory
        traj_pred.append(next_pose_pred)
        traj_gt.append(next_pose_gt)

        # Update the errors
        ATEs.append(ate(traj_gt[-1], traj_pred[-1]))
        cerror, direction_pred, direction_gt = ce(traj_gt[-1], traj_pred[-1])
        CEs.append(cerror)
        d_gt.append(direction_gt)
        d_pred.append(direction_pred)
        DEs.append(de(traj_gt[-1], traj_pred[-1]))
        RTEs.append(rte(T_gt, T_pred))
        ROTs.append(rot(T_gt, T_pred))

        # Update the translation
        rel_poses_pred.append(rel_pose_pred)
        rel_poses_gt.append(rel_pose_gt)
        Ts_pred.append(T_pred)
        Ts_gt.append(T_gt)

        # Show the process
        print('=== {}/{} trajectory position has been tested ==='.format(i+1, len(test_loader)))
        print('Error of this position >>>'.format(i, len(test_loader)))
        print('ATE: {:.4f} mm, CE: {:.4f}, DE: {:.4f} deg'.format(ATEs[-1], CEs[-1], DEs[-1]))
        print('Error of relative transition >>>')
        print('RTE: {:.4f} mm, ROT: {:.4f} deg'.format(RTEs[-1], ROTs[-1]))


    return np.array(ATEs), np.array(RTEs), np.array(ROTs), np.array(CEs), np.array(DEs), traj_pred, traj_gt, d_pred, d_gt, rel_poses_pred, rel_poses_gt, length


def save_results(save_path, obj, ates, rtes, rots, ces, des, traj_pred, traj_gt, d_pred, d_gt, rel_pred, rel_gt):
    os.mkdir(str(save_path) + '/results_{}'.format(obj))

    f = open(str(save_path) + '/results_{}'.format(obj) + '/ATEs.txt', 'w')
    for e in ates:
        f.write(str(e) + '\n')
    f.close()
    f = open(str(save_path) + '/results_{}'.format(obj) + '/RTEs.txt', 'w')
    for e in rtes:
        f.write(str(e) + '\n')
    f.close()
    f = open(str(save_path) + '/results_{}'.format(obj) + '/ROTs.txt', 'w')
    for e in rots:
        f.write(str(e) + '\n')
    f.close()
    f = open(str(save_path) + '/results_{}'.format(obj) + '/CEs.txt', 'w')
    for e in ces:
        f.write(str(e) + '\n')
    f.close()
    f = open(str(save_path) + '/results_{}'.format(obj) + '/DEs.txt', 'w')
    for e in des:
        f.write(str(e) + '\n')
    f.close()
    f = open(str(save_path) + '/results_{}'.format(obj) + '/traj_gt.txt', 'w')
    for pose in traj_gt:
        x = pose[0, -1]
        y = pose[1, -1]
        z = pose[2, -1]
        f.write(str(x) + ' ')
        f.write(str(y) + ' ')
        f.write(str(z) + '\n')
    f = open(str(save_path) + '/results_{}'.format(obj) + '/traj_pred.txt', 'w')
    for pose in traj_pred:
        x = pose[0, -1]
        y = pose[1, -1]
        z = pose[2, -1]
        f.write(str(x) + ' ')
        f.write(str(y) + ' ')
        f.write(str(z) + '\n')
    f.close()
    f = open(str(save_path) + '/results_{}'.format(obj) + '/directions_gt.txt', 'w')
    for d in d_gt:
        x = d[0]
        y = d[1]
        z = d[2]
        f.write(str(x) + ' ')
        f.write(str(y) + ' ')
        f.write(str(z) + '\n')
    f.close()
    f = open(str(save_path) + '/results_{}'.format(obj) + '/directions_pred.txt', 'w')
    for d in d_pred:
        x = d[0]
        y = d[1]
        z = d[2]
        f.write(str(x) + ' ')
        f.write(str(y) + ' ')
        f.write(str(z) + '\n')
    f.close()
    f = open(str(save_path) + '/results_{}'.format(obj) + '/T_gt.txt', 'w')
    for t in rel_gt:
        f.write(str(t) + '\n')
    f.close()
    f = open(str(save_path) + '/results_{}'.format(obj) + '/T_pred.txt', 'w')
    for t in rel_pred:
        f.write(str(t) + '\n')
    f.close()


def cal_distance(gt, pred):
    return math.sqrt((float(gt[0]) - float(pred[0])) ** 2 + (
                float(gt[1]) - float(pred[1])) ** 2 + (
                          float(gt[2]) - float(pred[2])) ** 2)


if __name__ == '__main__':
    main()


