# __author__ = haitham elmarakeby
from os import makedirs
from os.path import exists
import figure_3_classifier_heads as fig3
import figure_1_2_4 as fig124
import figure_4_samples_sizes as fig4
from config_path import PLOTS_PATH

if __name__ == '__main__':
    if not exists(PLOTS_PATH):
        makedirs(PLOTS_PATH)
    # Figure 1
    fig124.plot_figure1_compare_model_sizes(task='response', tuned=False)
    fig124.plot_figure1_compare_model_sizes(task='progression', tuned=False)

    # Figure 2
    fig124.plot_figure2_compare_domain_adaptation('response')
    fig124.plot_figure2_compare_domain_adaptation('progression')

    # Figure 4
    fig124.plot_figure4('response')
    fig124.plot_figure4('progression')

    #Figure 3
    fig3.plot_classifier(task = 'progression_tuned')
    fig3.plot_classifier(task = 'response_tuned')

    # Figure 4 (sample sizes)
    fig4.plot_originals()
    fig4.plot_tuned()

    fig4.plot_originals_progression()
    fig4.plot_tuned_progression()
