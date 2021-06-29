import torch
from torch import nn
from model import PredictionConvolutions, tiny_detector
from utils import gcxgcy_to_cxcy, cxcy_to_xy, find_jaccard_overlap, cxcy_to_gcxgcy, xy_to_cxcy
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import PascalVOCDataset


def main():
    device = 'cuda'
    n_classes = 21
    model = PredictionConvolutions(n_classes)
    # x = torch.randn(2, 3, 7, 7)
    # pool_conv5 = torch.randn(2, 512, 7, 7)
    # locs, class_scores = model(pool_conv5)
    # locs = locs.to(device)
    # print('locs:', locs.shape, locs)
    # print('class_scores', class_scores.shape, class_scores[0, 0, :])
    # predicted_scores = F.softmax(class_scores, dim=2)
    # print('predicted_scores.shape:', predicted_scores.shape)
    # # print('predicted_scores:',predicted_scores[0,0,:],predicted_scores[0,0,:].max())
    # # print('predicted_scores[i].max(dim=1):',predicted_scores[0],predicted_scores[0].max(dim=1))
    # class_score = predicted_scores[0][:, 0]
    # print('class_score:', class_score.shape, class_score)
    # score_above_min_threshold = class_score > 0.3
    # print('score_above_min_threshold:', score_above_min_threshold.shape, score_above_min_threshold)
    # n_score_above_threshold = score_above_min_threshold.sum().item()
    # print('n_score_above:', n_score_above_threshold)
    # class_score = class_score[score_above_min_threshold]
    # print('class_score', class_score)
    # class_score, sort_ind = class_score.sort(dim=0, descending=True)
    # print('class_score_after_sort:', class_score, sort_ind)
    Detector = tiny_detector(n_classes).to(device)
    # decode_locs = cxcy_to_xy(gcxgcy_to_cxcy(locs[0], Detector.create_prior_boxes()))
    # class_decoded_locs = decode_locs[score_above_min_threshold]
    # print('class_decoded_locs:', class_decoded_locs)
    # class_decoded_locs = class_decoded_locs[sort_ind]
    # print('class_decoded_locs1:', class_decoded_locs)
    # overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)
    # print('overlap:', overlap)
    # print('overlap.max(dim=0):', overlap.max(dim=0))
    # suppress = torch.zeros((n_score_above_threshold), dtype=torch.uint8).to(device)
    # print('suppress:', suppress)
    # for box in range(class_decoded_locs.size(0)):
    #     print('\noverlap[box]:', overlap[box], overlap[box] > 0.2)
    #     suppress = torch.max(suppress, (overlap[box] > 0.2).to(torch.uint8))
    #     suppress[box] = 0
    #     print('after each deal:', suppress)
    # print('\nsuppress_after_deal:', suppress)
    data_folder = "D:\Pycharm\\newproject\Object detection\json"
    batch_size = 16
    train_dataset = PascalVOCDataset(data_folder, split='train', keep_difficult=False)
    test_dataset = PascalVOCDataset(data_folder, split='test', keep_difficult=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn,
                             num_workers=0, pin_memory=True)
    for idx, (images, boxes, labels, _) in enumerate(train_loader):
            print('\nidx:',idx)
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            pred_locs, pred_cls = Detector(images)
            print('pred_cls:',pred_cls.shape)

            print('boxes:', boxes)
            true_class = torch.zeros((batch_size, 441), dtype=torch.long).to(device)
            true_locs = torch.zeros((batch_size, 441, 4), dtype=torch.float).to(device)
            for i in range(batch_size):
                print('i:\n',i)
                prior_cxcy = Detector.priors_cxcy
                n_priors = prior_cxcy.size(0)
                print('n_priors:',n_priors)
                print('boxes[i]:', boxes[i].shape, boxes[i])  # [n_objects,4]
                n_objects = boxes[i].size(0)
                print('n_objects:', n_objects)
                overlap = find_jaccard_overlap(boxes[i], prior_cxcy)
                print('overlap:', overlap.shape, overlap)  # [n_objects,441]
                # 行为图片中真实目标框数量，列为锚框数量，矩阵中第i行，第j列对应第i个真实框和第j个锚框之间的IOU值
                overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)
                print('overlap_for_each_prior', overlap_for_each_prior.shape, overlap_for_each_prior)  # [441]
                print('objects_for_each_prior', object_for_each_prior.shape, object_for_each_prior)  # [441]
                _, prior_for_each_object = overlap.max(dim=1)
                print('prior_for_each_object', prior_for_each_object)
                object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
                print('object_for_each_prior[prior_for_each_object]', object_for_each_prior[prior_for_each_object])
                print('object_for_each_prior_after_deal', object_for_each_prior)
                overlap_for_each_prior[prior_for_each_object] = 1.
                print('overlap_for_each_prior', overlap_for_each_prior)
                print('labels:', labels[i])
                label_for_each_prior = labels[i][object_for_each_prior]
                print('label_for_each_prior:', label_for_each_prior.shape, label_for_each_prior)
                label_for_each_prior[overlap_for_each_prior < 0.5] = 0
                print('label_for_each_prior_after_threshold:', label_for_each_prior)
                true_class[i] = label_for_each_prior
                true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), prior_cxcy)
                print('true_locs:', true_locs)
            positive_priors = true_class != 0
            print('positive_priors', positive_priors.shape, positive_priors)
            n_positive_priors = positive_priors.sum(dim=1)
            n_negtive_priors = 3 * n_positive_priors
            print('n_positive_priors', n_positive_priors)
            locs_criterion = nn.SmoothL1Loss()
            locs_loss = locs_criterion(pred_locs[positive_priors], true_locs[positive_priors])
            print('locs_loss:', locs_loss)
            cls_criterion = nn.CrossEntropyLoss(reduce=False)
            print('pred_cls:',pred_cls.shape)
            print('true_class:',true_class.shape)
            cls_loss = cls_criterion(pred_cls.view(-1, n_classes), true_class.view(-1))
            # 疑问，view之后两者size分别为[N*441,20],[N*441]这样不同size也可以进行CrossEntropyLoss吗
            print('cls_loss:', cls_loss)
            cls_loss = cls_loss.view(batch_size, n_priors)# [N,441]
            print('cls_loss_after_view:', cls_loss.shape,cls_loss)  # [N*441]

            cls_loss_pos = cls_loss[positive_priors]
            print('cls_loss_pos:', cls_loss_pos)
            cls_loss_neg = cls_loss.clone()  # [N,441]
            print('cls_loss_neg:', cls_loss_neg.shape, cls_loss_neg)
            cls_loss_neg[positive_priors] = 0
            print('cls_loss_neg_after_deal;:', cls_loss_neg)
            cls_loss_neg, _ = cls_loss_neg.sort(dim=1, descending=True)  # [N,441]
            print('cls_loss_neg_after_sort:', cls_loss_neg)
            hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(cls_loss_neg).to(device)
            print('hardness_ranks:', hardness_ranks.shape)
            print(hardness_ranks)
            print('n_negtive_priors:', n_negtive_priors)
            hard_negatives = hardness_ranks < n_negtive_priors.unsqueeze(1)  # [N,441]
            print('hard_negatives:', hard_negatives.shape, hard_negatives)
            cls_loss_neg_hard = cls_loss_neg[hard_negatives]
            print('cls_loss_neg_hard:', cls_loss_neg_hard.shape, cls_loss_neg_hard)
            print('cls_loss_pos:', cls_loss_pos.shape, cls_loss_pos)
            conf_loss = (cls_loss_neg_hard.sum() + cls_loss_pos.sum()) / n_positive_priors.sum().float()
            print('n_p_s_f:',n_positive_priors.sum().float())
            print('cls_loss_neg_hard.sum():', cls_loss_neg_hard.sum())
            print('cls_loss_pos.sum():', cls_loss_pos.sum())
            print('conf_loss:', conf_loss)
            print('--------------------------------------------------------------------------------------------------------------------------------\n')


if __name__ == '__main__':
    main()
