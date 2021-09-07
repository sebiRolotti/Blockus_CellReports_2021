import matplotlib as mpl
mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

import mini_repo.analysis.place_cell_analysis as pca
from mini_repo.classes.dbclasses import dbMouse
from mini_repo.classes.classes import pcExperimentGroup

import numpy as np


def savefigs(pdf_pages, figs):
    """Save a single figure or list of figures to a multi-page PDF.

    This function is mostly used so that the same call can be used for a single
    page or multiple pages. Will close Figures once they are written.

    Parameters
    ----------
    pdf_pages : matplotlib.backends.backend_pdf.PdfPages
        PdfPage instance that the figures will get written to.
    figs : matplotlib.pyplot.Figure or iterable of Figures

    """
    try:
        for fig in figs:
            pdf_pages.savefig(fig)
            plt.close(fig)
    except TypeError:
        pdf_pages.savefig(figs)
        plt.close(figs)


def sorted_heatmaps(grp):

    fig = plt.figure()

    n_days = len(grp)
    tcs = grp.data()

    for i in xrange(n_days):

        # try:
        sortby_expt = grp[i]
        centroids = pca.calcCentroids(tcs[sortby_expt], grp.pfs()[sortby_expt], returnAll=True)
        # except AssertionError:
        #     continue
        centroids = [c[0] if len(c) >= 1 else 100 for c in centroids]
        sort_idx = np.argsort(centroids)

        for j in xrange(n_days):

            fig_idx = i * n_days + j + 1

            ax = fig.add_subplot(n_days, n_days, fig_idx)

            sns.heatmap(tcs[grp[j]][sort_idx, :], ax=ax, cbar=False, cmap=sns.cm.rocket, rasterized=True)

            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title('{}'.format(grp[j].trial_id))

    fig.suptitle('{} FOV {}'.format(grp[0].parent.mouse_name, grp[0].get('uniqueLocationKey')))

    return fig


def all_pfs(grp, save_name):

    tcs = grp.data()
    pfs = grp.pfs()

    all_tcs = np.vstack([tcs[e] for e in grp])

    all_tcs = all_tcs / np.nanmax(all_tcs, axis=1, keepdims=True)

    all_centroids = []

    for expt in grp:
        centroids = pca.calcCentroids(tcs[expt], pfs[expt], returnAll=True)
        centroids = [c[0] if len(c) >= 1 else 100 for c in centroids]
        all_centroids.extend(centroids)

    sort_idx = np.argsort(all_centroids)

    fig = plt.figure(figsize=(10, 100))
    ax = fig.add_subplot(111)

    sorted_tcs = all_tcs[sort_idx, :]
    sorted_tcs = np.nan_to_num(sorted_tcs)

    sns.heatmap(sorted_tcs, ax=ax, cbar=False, cmap=sns.cm.rocket, rasterized=True)

    ax.set_xticks([])
    ax.set_yticks([ax.get_yticks()[0], ax.get_yticks()[-1]])
    ax.set_yticklabels([0, all_tcs.shape[0] - 1], rotation=0)

    # ax.set_ylim([0, all_tcs.shape[0]])
    # ax.set_yticks([0, all_tcs.shape[0] - 1])

    fig.savefig(save_name)


def plot_pfs(grp, pp):

    n_days = 4

    for i in xrange(0, len(grp), n_days):

        fig = sorted_heatmaps(grp.subGroup(grp[i:i + n_days]))

        savefigs(pp, fig)
        plt.close(fig)


if __name__ == '__main__':

    # CHANGE THIS
    save_path = '.'

    mice = ['hb169_2', 'hb177_2', 'hb184_3', 'hb189_1', 'hb190_1', 'hb192_1']
    expts = []
    for mouse in mice:
        expts.extend(dbMouse(mouse).imagingExperiments())

    expts = [e for e in expts if 'GOL' not in e.get('tSeriesDirectory')]

    bad_tseries = ['/data2/Heike/hb177_2/190627/hb177_2_RF_FOV2_day2-001']
    expts = [e for e in expts if not np.any([x in e.get('tSeriesDirectory') for x in bad_tseries])]

    expts = sorted(expts, key=lambda x: (x.parent.mouse_name, x.uniqueLocationKey, x.day))

    green_kwargs = {'imaging_label': 'drawn_green',
                    'nPositionBins': 100,
                    'channel': 'Ch2',
                    'demixed': False,
                    'pf_subset': None,
                    'signal': 'spikes'}

    red_kwargs = {'imaging_label': 'drawn_red',
                  'nPositionBins': 100,
                  'channel': 'Ch2',
                  'demixed': False,
                  'pf_subset': None,
                  'signal': 'spikes'}

    wt_grp = pcExperimentGroup(expts, **green_kwargs)
    ko_grp = pcExperimentGroup(expts, **red_kwargs)

    # pp = PdfPages('/home/sebi/plots/robo_wt_pfs.pdf')
    # plot_pfs(wt_grp, pp)
    # pp.close()

    # pp = PdfPages('/home/sebi/plots/robo_ko_pfs.pdf')
    # plot_pfs(ko_grp, pp)
    # pp.close()


    all_pfs(wt_grp, os.path.join(save_path, 'robo_wt_all_pfs.pdf'))
    all_pfs(ko_grp, os.path.join(save_path, 'robo_ko_all_pfs.pdf'))
