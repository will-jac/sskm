import numpy as np

import argparse

# ML methods
import models
import tests

def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='SS KM', fromfile_prefix_chars='@')
    parser.add_argument('-method', type=str, default='coreg', help='kernel method to use')
    parser.add_argument('-test', type=str, default='sphere', help='test to use')
    parser.add_argument('-kernel', type=str, default='rbf', help='kernel function to use')
    parser.add_argument('-gamma', type=float, default=1.0, help='gamma for pairwise kernel')
    parser.add_argument('-degree', type=float, default=3.0, help='degree for pairwise kernel')
    parser.add_argument('-coef0', type=float, default=1.0, help='coef0 for pairwise kernel')
    parser.add_argument('-l2', type=float, default=0.1, help='l2 coefficient')
    parser.add_argument('-manifold', type=float, default=0.1, help='manifold coefficient')
    parser.add_argument('-weight', type=str, default='gaussian', help='weight for manifold')
    parser.add_argument('-mu', type=float, default=0.1, help='difference coefficient')
    parser.add_argument('-norm', type=float, default=0.1, help='norm coefficient (gamma in coreg)')
    parser.add_argument('-k', type=float, default=0.1, help='number of kNN')
    parser.add_argument('-p', type=int, default=2, help='laplacian exponent')

    return parser

def execute_exp(args):

    if args.model in models.models:
        model = models.models[args.model]
    else:
        print('error: model', args.model, 'not found') 
        return
    
    if args.test in tests.tests:
        test = tests.tests[args.test]
    else:
        print('error: test', args.test, 'not found')
        return

    # construct the model
    model = model(
        kernel=args.kernel, 
        gamma=args.gamma,
        degree=args.degree,
        coef0=args.coef0,
        l2=args.l2,
        manifold=args.manifold,
        mu=args.mu,
        g=args.norm,
        kNN=args.k,
        p=args.p,
        weight=args.weight)

    # run the tets
    results = test(model)

    # save the output
    fname = args.test + '_' + args.method + \
            '_l2_' + args.l2 + '_man_' + args.manifold + \
            '_mu_' + args.mu + '_norm_' + args.norm + \
            '_k_' + args.k + '_p_' + args.p + \
            '_weight_' + args.weight + '.out'

    with open(fname, 'w') as f:
        f.write(str(results))

if __name__ == "__main__":
# if True:
    parser = create_parser()
    args = parser.parse_args()
    execute_exp(args)


    # svm = SVC()
    # svm.name = 'SVM'

    # models = [
    #     # rff(D=10),
    #     # rff(D=100),
    #     # rff(D=1000),
    #     # Nystrom_Ridge(10),
    #     # Nystrom_Ridge(100),
    #     # Nystrom_Ridge(1000),
    #     # svm,
    #     # kernel.LS(),
    #     # kernel.KLS('rbf'),
    #     # kernel.RLSKernel('rbf', 1.0, True, True),
    #     # kernel.RLSKernel('rbf', 1.0, False, True),

    #     # kernel.RidgeKernel('rbf', 1.0),

    #     # kernel.RLSKernel('rbf', 1.0, True, True),
    #     ManifoldRLS('rbf', 0.1, kNN=2),

    #     # SSManifoldRLS('rbf', 0.01),
    #     # SSLapRLS('rbf', 0, 0.01),
    #     SSCoMR('rbf', g=1, L2_coef=0.001, manifold_coef=0.1, mu=0.1),
    #     # SSCoReg('rbf', g=0.001, L2_coef=10, manifold_coef=10, mu=0.1),
    #     # SSCoReg('rbf', g=0.001, L2_coef=10, manifold_coef=1, mu=0.01),
    #     # SSCoReg('rbf', g=0.001, L2_coef=10, manifold_coef=0.1, mu=0.01),

    #     # SSCoReg('rbf', g=0.001, L2_coef=10, manifold_coef=100, mu=0.01),
    #     # SSCoReg('rbf', g=0.01, L2_coef=1, manifold_coef=10, mu=0.01),
    #     # SSCoReg('rbf', g=0.001, L2_coef=10, manifold_coef=1, mu=1),
    #     # SSCoReg('rbf', g=0.001, L2_coef=10, manifold_coef=10, mu=1),
    #     SSCoReg('rbf', g=0.001, L2_coef=10, manifold_coef=100, mu=1),
    #     SSCoRegSolver('rbf', g=0.001, L2_coef=10, manifold_coef=100, mu=1),
    # ]
    # tests = [
    #     lambda model : sphere_test(model, n=100, d=2, u=100, show_plots=False),
    #     # lambda model : checkerboard_test(model, seed=1, noise=0.0),
    #     # lambda model : checkerboard_test(model, seed=1, noise=0.2),
    #     # lambda model : adult_test(model)
    # ]
    # test_names = [
    #     'sphere',
    #     # 'checkerboard', 'checkerboard_noise',
    #     'adult'
    # ]

    # # X, y = checkerboard.generate_data((100000, 2), noise=0.1, seed=1, shuffle=False)
    # # labels = ['#1f77b4' if abs(l) < 0.5 else '#ff7f0e' for l in y]
    # # plt.scatter(X[:,0], X[:,1], c=labels)
    # # plt.show()

    # test_runner(models, tests, test_names)

    # [draw_decision_boundary(m) for m in models]

# def create_parser():
#     parser = argparse.ArgumentParser(description='Kernel Method Bake-Off')

#     parser.add_argument('--model', '-m', type=str, default='rff')
#     parser.add_argument('--dataset', '-d', type=str, default='adult')
#     parser.add_argument('--kernel', '-k', type=str, default='gaussian')
#     parser.add_argument('--Dimensions', '-D', type=int, default=100)
#     parser.add_argument('--n_samples', '-n', type=int, default=1000)
#     parser.add_argument('--C', '-c', type=float, default=1.0)

#     return parser

# def execute(args):
#     m = args.model
#     if m == 'rff':
#         model = rff(D = args.Dimensions, k = args.kernel)
#     if m == 'SVC':
#         model = SVC()
#     if m == 'SVM':
#         model = SVM(kernel = args.kernel, c = args.C)

#     if args.dataset == 'adult':
#         adult_test([model], [args.model])
#     elif args.dataset == 'sphere':
        # sphere_test([model], [args.model], show_plots=True)

