import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated argument

    parser.add_argument('--FL_name', type=str, default='bm-fl', help="FL name  bm-fl")
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
   
    parser.add_argument('--train_bs', type=int, default=64, help="client batch_size")
    parser.add_argument('--train_ep', type=int, default=5, help="client epoch")
    parser.add_argument('--w_epochs', type=int, default=4, help="weight update epoch")
    parser.add_argument('--epoch', type=int, default=70, help="fl epoch")

    # other
    parser.add_argument('--model', type=str, default='BC-GAN', help="model name")
    parser.add_argument('--dataset', type=str, default='mnist', help="dataset name mnist,8fashion,cifar_10")
    parser.add_argument('--num_classes', type=int, default=10, help="class num")
    parser.add_argument('--num_clients', type=int, default=5, help="client num")
    parser.add_argument('--num_its', type=int, default=2, help="ITS num")
    

    #bm_fl
    parser.add_argument("--pre_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument('--lr', type=float, default=0.001, help="learning_rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--wlow", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--wup", type=float, default=1.0, help="adam: decay of first order momentum of gradient")
    
    parser.add_argument('--slidL', type=int, default=3, help="slid length")
    parser.add_argument("--unsample_num", type=int, default=2, help="unsample of unsample")
    parser.add_argument("--epsilon", type=int, default=0, help="epsilon set")

    args = parser.parse_args()
    print('config')
    return args

