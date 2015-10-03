import glob, itertools, argparse
from collections import OrderedDict

import numpy as np
import scipy.io
import pandas
import seaborn as sns
import networkx as nx

sns.set_style('dark')
cmap = sns.cubehelix_palette(as_cmap=True)
sns.set_palette("Set2")
# cmap = sns.color_palette("RdBu_r", 7, as_cmap=True)

ROIS = ['V1', 'lo', 'pfs', 'tos', 'ppa', 'rsc', 'ofa', 'ffa', 'sts', 'eba']

def _hemi(hemi):
    if hemi == 'rh':
        h = 'r'
        hi = 0
    else:
        h = 'l'
        hi = 1
    return h, hi

def read_roi(roi='sts', hemi='rh'):
    h, hi = _hemi(hemi)
    try:
        roi_file = glob.glob('../roi_grids/%s*%s_grid.mat' % (h, roi))[0]
    except:
        print 'no', roi
    else:
        roi_data = scipy.io.loadmat(roi_file)
        return roi_data['grid_roi_thresh'][0,hi]

def get_rois(hemi='rh', plot=False):
    all_rois = []
    for roi in ROIS:
        roi_data = read_roi(roi=roi, hemi=hemi)
        if roi_data is not None:
            all_rois.append(roi_data)
        if plot:
            sns.plt.imshow(roi_data['grid_roi_thresh'][0,hi])
            sns.plt.title(roi)
            sns.plt.show()
    all_rois = np.array(all_rois)
    sns.plt.imshow(np.max(all_rois,0))
    sns.plt.show()
    return all_rois

def pred_map(hemi='rh'):
    all_rois = get_rois(hemi=hemi)
    preds = scipy.io.loadmat('../data/%s_model.mat' % MODEL)
    rpreds = preds['%s_stats' % hemi][-1,0]
    print np.min(rpreds), np.max(rpreds)
    rpreds[np.isnan(all_rois[0])] = np.nan
    sns.plt.imshow(rpreds, cmap='RdBu_r', vmin=-.8, vmax=.8)
    sns.plt.colorbar()
    sns.plt.show()

def tuning(hemi='rh', roi='sts', cluster=4):
    df = pandas.read_csv('../analyses/tuning_curves_sts.csv')
    df = df[df.cluster==cluster]
    np.random.seed(0)

    for g, genre in enumerate(df.genre.unique()):
        sel = df[df.genre==genre]
        G = nx.star_graph(len(sel))
        pos = nx.spring_layout(G)
        colors = np.sign(sel.weight).astype(int).tolist()
        node_size = 2000*np.abs(sel.weight)
        node_size = [np.mean(node_size)] + node_size.tolist()
        cmap = sns.color_palette('Set2', 2)
        cond = [np.mean(sel.weight)] + sel.weight.tolist()
        colors = [cmap[0] if c<0 else cmap[1] for c in cond]

        labels = dict([(k+1,v) for k,v in enumerate(sel.style)])
        labels[0] = genre

        sns.plt.subplot(2,3,g+1)
        nx.draw(G, pos, node_color=colors, edge_color='gray',
                width=1, with_labels=True,
                node_size=node_size, labels=labels)

    sns.plt.savefig('../images/%s_%s_%s.svg' % (hemi, roi, cluster))
    sns.plt.show()

def tuning_scale(**kwargs):
    sizes = 2000 * np.array([-1.5, -1.2, -.9, -.6, -.3, 0, .3, .6])
    G = nx.path_graph(len(sizes))
    pos = nx.spring_layout(G)

    cmap = sns.color_palette('Set2', 2)
    # cond = [np.mean(sel.weight)] + sel.weight.tolist()
    colors = [cmap[0] if c<0 else cmap[1] for c in sizes]

    nx.draw(G, pos, node_color=colors, edge_color='gray',
            width=1, with_labels=False,
            node_size=np.abs(sizes))
    sns.plt.savefig('../images/tuning_scale.svg')
    sns.plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('func')
parser.add_argument('-m', '--model', choices=['flickr','subitizing','scene','object', 'jonas_object'])
parser.add_argument('-e', '--hemisphere', choices=['rh','lh'])
args = parser.parse_args()

MODEL = args.model
eval(args.func)(hemi=args.hemisphere)
