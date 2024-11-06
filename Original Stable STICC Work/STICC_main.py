from STICC_solver import STICC
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parameters of the STICC')
parser.add_argument('--fname', type=str,
                    default="synthetic_data.txt", help='Input data name')
parser.add_argument('--oname', type=str,
                    default="result_synthetic_data.txt", help='Output file name')
parser.add_argument('--attr_idx_start', type=int,
                    default=1, help='Attribute start index')
parser.add_argument('--attr_idx_end', type=int,
                    default=5, help='Attribute end index')
parser.add_argument('--spatial_idx_start', type=int,
                    default=6, help='Neighbouring object start index')
parser.add_argument('--spatial_idx_end', type=int, default=8,
                    help='Neighbouring object end index')
parser.add_argument('--coord_idx_start', type=int,
                    default=6, help='Raw coordinates start index')
parser.add_argument('--coord_idx_end', type=int, default=8,
                    help='Raw coordinates end index')
parser.add_argument('--spatial_radius', type=int,
                    default=3, help='Radius of the subregion')
parser.add_argument('--number_of_clusters', type=int,
                    default=5, help='Number of clusters')
parser.add_argument('--lambda_parameter', type=float,
                    default=0.1, help='Lambda')
parser.add_argument('--beta', type=float, default=5, help='Beta')
parser.add_argument('--maxIters', type=int, default=20, help='Max Iterations')

parser.add_argument('--method', type=str, default='glasso', help='Choose method')
parser.add_argument('--mask_rate', type=float, default='0.0', help='Dynamic masking rate')
parser.add_argument('--local_radius', type=int, default='3', help='Local mean radius')
parser.add_argument('--init_random', action='store_true', help='Toggle initial random imputation')
parser.add_argument('--interp', type=str, default='missglasso', help='Imputation method')
parser.add_argument('--init_thres', type=float, default=5.68, help='Initial Mahalanobis threshold')
parser.add_argument('--mala_thres', type=float, default=5.68, help='Mahalanobis threshold')
parser.add_argument('--randomize', type=str, default="None", help='Randomization choice')

args = parser.parse_args()

def main():
    sticc = STICC(spatial_radius=args.spatial_radius, number_of_clusters=args.number_of_clusters,
             lambda_parameter=args.lambda_parameter, beta=args.beta, maxIters=args.maxIters,
             threshold=2e-5, write_out_file=False, prefix_string="output_folder/", num_proc=1,
             attr_idx_start=args.attr_idx_start, attr_idx_end=args.attr_idx_end,
             coord_idx_start=args.coord_idx_start, coord_idx_end=args.coord_idx_end,
             spatial_idx_start=args.spatial_idx_start, spatial_idx_end=args.spatial_idx_end)

    if args.method == 'glasso':
        (cluster_assignment, cluster_MRFs) = sticc.fit(input_file=args.fname)
    elif args.method == 'missglasso':
        (cluster_assignment, cluster_MRFs) = sticc.fit_missglasso(input_file=args.fname, radius=args.local_radius)
    elif args.method == 'missglasso_missing':
        (cluster_assignment, cluster_MRFs) = sticc.fit_missglasso_static_missing(input_file=args.fname, interp=args.interp, radius=args.local_radius)
    elif args.method == 'missglasso_static':
        (cluster_assignment, cluster_MRFs) = sticc.fit_missglasso_static(input_file=args.fname, interp=args.interp, radius=args.local_radius, thres=args.mala_thres, rate=args.mask_rate, randomize=args.randomize)
    elif args.method == 'missglasso_dynamic':
        (cluster_assignment, cluster_MRFs) = sticc.fit_missglasso_dynamic(input_file=args.fname, radius=args.local_radius, initial_thres=args.init_thres, thres=args.mala_thres, rate=args.mask_rate, randomize=args.randomize)
    elif args.method == 'missglasso_mixed':
        (cluster_assignment, cluster_MRFs) = sticc.fit_missglasso_mixed(input_file=args.fname, radius=args.local_radius, thres=args.mala_thres, rate=args.mask_rate)
    elif args.method == 'globalmean':
        (cluster_assignment, cluster_MRFs) = sticc.fit_global_mean(input_file=args.fname)
    elif args.method == 'localmean':
        (cluster_assignment, cluster_MRFs) = sticc.fit_local_mean(input_file=args.fname, radius=args.local_radius)
    elif args.method == 'kriging':
        (cluster_assignment, cluster_MRFs) = sticc.fit_kriging(input_file=args.fname)
    elif args.method == 'tree':
        (cluster_assignment, cluster_MRFs) = sticc.fit_tree(input_file=args.fname, radius=args.local_radius)
    else:
        assert False, "No supported method!"

    # Save cluster output
    print(cluster_assignment)
    np.savetxt(args.oname, cluster_assignment, fmt='%d', delimiter=',')

    # Save MRF as npy
    for key, value in cluster_MRFs.items():
        with open(f'output_folder/MRF_{args.fname.split(".")[0]}_{key}.npy', 'wb') as f:
            np.save(f, np.array(value))

if __name__ == '__main__':
    main()