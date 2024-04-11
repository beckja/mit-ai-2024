from matplotlib import pyplot as plt
from matplotlib import patches


def plot_timeline(gt_object, p_object, title_info, tolerance):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    plot_type_timeline(gt_object, p_object, ax1, 'EW', tolerance)
    plot_type_timeline(gt_object, p_object, ax2, 'NS', tolerance)
    plt.xlabel('TimeIndex')
    fig.suptitle(title_info, fontsize=10)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    legend_elements = [
        plt.Line2D([0], [0], color='green', linestyle='dashed', label='True Positive (TP)'),
        plt.Line2D([0], [0], color='blue', linestyle='dashed', label='False Positive (FP)'),
        plt.Line2D([0], [0], color='red', linestyle='dashed', label='False Negative (FN)'),
        patches.Patch(color='grey', alpha=0.2, label='Tolerance Interval')
    ]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4)
    plt.show()


def plot_type_timeline(gt_object, p_object, ax, type_label, tolerance):
    '''Plot the node detection timeline for comparison of truth and prediction.

    type_label is 'EW' or 'NS'.
    tolerance is the number of timesteps allowed for correlation of truth
      and prediction.
    '''
    ground_truth_type = gt_object[gt_object['Direction'] == type_label]
    participant_type = p_object[p_object['Direction'] == type_label]
    for _, row in ground_truth_type.iterrows():
        label = row['Node'] + '-' + row['Type']
        ax.scatter(row['TimeIndex'], 2, color='black')
        ax.text(row['TimeIndex'] + 3, 2.05, label, rotation=45)
        ax.fill_betweenx([1, 2], row['TimeIndex'] - tolerance,
                            row['TimeIndex'] + tolerance, color='grey',
                            alpha=0.2)
        if row['classification'] == 'TP':
            ax.text(row['TimeIndex'] + tolerance + .5, 1.5, 
                    str(row['distance']), 
                    color='black')
            ax.plot([row['TimeIndex'], 
                        row['TimeIndex'] + row['distance']], [2, 1], 
                        color='green', linestyle='dashed')
        elif row['classification'] == 'FP':
            ax.plot([row['TimeIndex'], 
                        row['TimeIndex'] + row['distance']], [2, 1], 
                        color='blue', linestyle='dashed')
        elif row['classification'] == 'FN':
            ax.plot([row['TimeIndex'], 
                        row['TimeIndex']], [2, 2.2], color='red', 
                        linestyle='dashed')

    for _, row in participant_type.iterrows():
        label = row['Node'] + '-' + row['Type']
        ax.scatter(row['TimeIndex'], 1, color='black')
        ax.text(row['TimeIndex'] + 3, 1.05, label, rotation=45)
        if row['classification'] == 'FP' and row['matched'] == False:
            ax.plot([row['TimeIndex'], row['TimeIndex']], [1, 0.8], 
                    color='blue', linestyle='dashed')

    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.grid(True)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Participant', 'Ground truth'])
    ax.set_title(type_label)
