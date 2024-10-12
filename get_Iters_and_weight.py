import torch
from Data_preprocessing import all_dataCreateDatasetIters,cal_cls_fusion_factor


def get_iters_and_weight(source1_dir,source2_dir,source3_dir,target_dir,batch_size=32,norm=True,cuda=True):
     
    print('*** dataset setting***')
    print('source 1 : ', source1_dir.split('/')[-2])
    print('source 2 : ', source2_dir.split('/')[-2])
    print('source 3 : ', source3_dir.split('/')[-2])
    print('target   : ', target_dir.split('/')[-2])
    print()

    source1_loader = all_dataCreateDatasetIters(source1_dir, requried_Val_iter=False, norm=norm,
                                        batch_size=batch_size)
    source2_loader = all_dataCreateDatasetIters(source2_dir, requried_Val_iter=False, norm=norm,
                                        batch_size=batch_size)
    source3_loader = all_dataCreateDatasetIters(source3_dir, requried_Val_iter=False, norm=norm,
                                        batch_size=batch_size)

    target_loader = all_dataCreateDatasetIters(target_dir, norm=norm, batch_size=batch_size)

    weight = cal_cls_fusion_factor([source1_loader, source2_loader, source3_loader])
    if cuda:
        weight = weight.cuda()
    return source1_loader,source2_loader,source3_loader,target_loader,weight
