# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='demo/vg1.jpg',
                        help="Path of the test image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_entities', default=100, type=int)
    parser.add_argument('--num_triplets', default=200, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')

    # Device (CPU / CUDA)
    parser.add_argument('--device', default='cpu',
                        help='device to use for inference: cpu or cuda')

    parser.add_argument('--resume', default='ckpt/checkpoint0149_oi.pth')
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--set_iou_threshold', default=0.7, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)

    parser.add_argument('--return_interm_layers', action='store_true')
    return parser


def main(args):

    # -------- Device handling --------
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # -------- Image transform --------
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    # -------- Bounding box utils --------
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        return torch.stack([
            x_c - 0.5 * w,
            y_c - 0.5 * h,
            x_c + 0.5 * w,
            y_c + 0.5 * h
        ], dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        scale = torch.tensor([img_w, img_h, img_w, img_h],
                             dtype=torch.float32,
                             device=b.device)
        return b * scale

    # -------- Classes --------
    CLASSES = [
        'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach',
        'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot',
        'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet',
        'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow',
        'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant',
        'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food',
        'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair',
        'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house',
        'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg',
        'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain',
        'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
        'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant',
        'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock',
        'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe',
        'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier',
        'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table',
        'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track',
        'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable',
        'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire',
        'woman', 'zebra'
    ]

    REL_CLASSES = [
        '__background__', 'above', 'across', 'against', 'along', 'and', 'at',
        'attached to', 'behind', 'belonging to', 'between', 'carrying',
        'covered in', 'covering', 'eating', 'flying in', 'for', 'from',
        'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of',
        'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near',
        'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of',
        'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under',
        'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears',
        'with'
    ]

    # -------- Build model --------
    model, _, _ = build_model(args)
    model.to(device)

    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # -------- Load image --------
    im = Image.open(args.img_path).convert("RGB")
    img = transform(im).unsqueeze(0).to(device)

    # -------- Inference --------
    with torch.no_grad():
        outputs = model(img)

    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]

    keep = (
        (probas.max(-1).values > 0.3) &
        (probas_sub.max(-1).values > 0.3) &
        (probas_obj.max(-1).values > 0.3)
    )

    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    topk = 10
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    scores = (
        probas[keep_queries].max(-1)[0] *
        probas_sub[keep_queries].max(-1)[0] *
        probas_obj[keep_queries].max(-1)[0]
    )
    indices = torch.argsort(-scores)[:topk]
    keep_queries = keep_queries[indices]

# -------- Visualization --------
    fig, axs = plt.subplots(ncols=len(indices), nrows=1, figsize=(22, 7))
    if len(indices) == 1:
        axs = [axs]
    
    for idx, ax, sb, ob in zip(
        keep_queries, axs,
        sub_bboxes_scaled[indices],
        obj_bboxes_scaled[indices]
    ):
        ax.imshow(im)
        ax.add_patch(plt.Rectangle(
            (sb[0], sb[1]), sb[2] - sb[0], sb[3] - sb[1],
            fill=False, color='blue', linewidth=2
        ))
        ax.add_patch(plt.Rectangle(
            (ob[0], ob[1]), ob[2] - ob[0], ob[3] - ob[1],
            fill=False, color='orange', linewidth=2
        ))
        title = (
            f"{CLASSES[probas_sub[idx].argmax()]} "
            f"{REL_CLASSES[probas[idx].argmax()]} "
            f"{CLASSES[probas_obj[idx].argmax()]}"
        )
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("reltr_output.png", dpi=150)
    plt.close()
    print("Saved output to reltr_output.png")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'RelTR inference',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
