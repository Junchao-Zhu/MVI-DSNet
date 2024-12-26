import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # Number of filter elements (must be odd)
    p = np.ones(nf // 2)  # Ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def plot_mc_curve(px, py, names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # Display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # Plot (confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # Plot (confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f'{ylabel}-Confidence Curve')
    plt.show()

    # fig.savefig(save_dir, dpi=250)
    # plt.close(fig)


def draw_f1_curve(file_dir, xlabel='Confidence', ylabel='F1'):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    files = os.listdir(file_dir)
    # files.sort()  # Sort alphabetically first
    files.sort(key=lambda x: (x.startswith('Ours'), x != 'Ours'))

    if 'Ours' in files:
        files.remove('Ours')
        files.append('Ours')

    num_files = len(files)
    color = plt.cm.viridis(np.linspace(0, 1, num_files))

    for i, file in enumerate(files):
        px = np.load(os.path.join(file_dir, file, 'px.npy'))
        f1 = np.load(os.path.join(file_dir, file, 'f1.npy'))

        y = smooth(f1.mean(0), 0.05)
        if file == 'Ours':
            ax.plot(px, y, linewidth=1.75, color='red', label=f'{file:} {y.max():.2f} at {px[y.argmax()]:.3f}')
        else:
            ax.plot(px, y, linewidth=1.75, color=color[i], label=f'{file:} {y.max():.2f} at {px[y.argmax()]:.3f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f'{ylabel}-Confidence Curve')
    plt.show()


def draw_R_curve(file_dir, xlabel='Confidence', ylabel='Recall'):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    files = os.listdir(file_dir)
    # files.sort()  # Sort alphabetically first
    files.sort(key=lambda x: (x.startswith('Ours'), x != 'Ours'))
    if 'Ours' in files:
        files.remove('Ours')
        files.append('Ours')

    num_files = len(files)
    color = plt.cm.viridis(np.linspace(0, 1, num_files))

    for i, file in enumerate(files):
        px = np.load(os.path.join(file_dir, file, 'px.npy'))
        f1 = np.load(os.path.join(file_dir, file, 'r.npy'))

        y = smooth(f1.mean(0), 0.05)
        if file == 'Ours':
            ax.plot(px, y, linewidth=1.75, color='red', label=f'{file:} {y.max():.2f} at {px[y.argmax()]:.3f}')
        else:
            ax.plot(px, y, linewidth=1.75, color=color[i], label=f'{file:} {y.max():.2f} at {px[y.argmax()]:.3f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f'{ylabel}-Confidence Curve')
    plt.show()


def draw_P_curve(file_dir, xlabel='Confidence', ylabel='Precision'):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    files = os.listdir(file_dir)
    # files.sort()  # Sort alphabetically first
    files.sort(key=lambda x: (x.startswith('Ours'), x != 'Ours'))
    if 'Ours' in files:
        files.remove('Ours')
        files.append('Ours')

    num_files = len(files)
    color = plt.cm.viridis(np.linspace(0, 1, num_files))

    for i, file in enumerate(files):
        px = np.load(os.path.join(file_dir, file, 'px.npy'))
        f1 = np.load(os.path.join(file_dir, file, 'p.npy'))

        y = smooth(f1.mean(0), 0.05)
        if file == 'Ours':
            ax.plot(px, y, linewidth=1.75, color='red', label=f'{file:} {y.max():.2f} at {px[y.argmax()]:.3f}')
        else:
            ax.plot(px, y, linewidth=1.75, color=color[i], label=f'{file:} {y.max():.2f} at {px[y.argmax()]:.3f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f'{ylabel}-Confidence Curve')
    plt.show()


def plot_pr_curve(file_dir):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    files = os.listdir(file_dir)
    # files.sort()  # Sort alphabetically first
    files.sort(key=lambda x: (x.startswith('Ours'), x != 'Ours'))
    if 'Ours' in files:
        files.remove('Ours')
        files.append('Ours')

    num_files = len(files)
    color = plt.cm.viridis(np.linspace(0, 1, num_files))

    for i, file in enumerate(files):
        px = np.load(os.path.join(file_dir, file, 'px.npy'))
        py = np.load(os.path.join(file_dir, file, 'py.npy'))
        ap = np.load(os.path.join(file_dir, file, 'ap.npy'))
        py = np.stack(py, axis=1)
        if file == 'Ours':
            ax.plot(px, py.mean(1), linewidth=1.75, color='red', label=f'{file} %.3f mAP@0.5' % ap[:, 0].mean())
        else:
            ax.plot(px, py.mean(1), linewidth=1.75, color=color[i], label=f'{file} %.3f mAP@0.5' % ap[:, 0].mean())

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title('Precision-Recall Curve')
    plt.show()


def confusion_matrix(file_dir):
    # File path
    for file in os.listdir(file_dir):
        print(file)
        file_name = file

        file_path = os.path.join(file_dir, file, 'data.txt')

        # Read data from file
        with open(file_path, 'r') as file:
            data = file.read().strip()
            values = data.split(',')
            tp, fp, fn, tn = map(float, values)

        # Create a 2x2 confusion matrix
        confusion_matrix = np.array([[tp, fp],
                                     [fn, tn]])

        # Set labels
        labels = ['MVI', 'Background']
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap='Blues', ax=ax, cbar=False,
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": 24, 'fontweight': 'bold'})

        ax.tick_params(axis='both', which='major', labelsize=22)  # Modify the font size of major tick labels

        # Set title and axis labels
        ax.set_title(f'{file_name}', fontsize=26, fontweight='bold')
        # ax.set_xlabel('True Label', fontsize=22)
        ax.set_ylabel('Predicted Label', fontsize=22)

        save_path = os.path.join(r'E:\毕业设计\new_exps\confusion-matrix\comparison', f'{file_name}.png')
        # print(save_path)

        # # Show the heatmap
        plt.savefig(save_path)


def final_confusion(file_dir):
    files = os.listdir(file_dir)
    files.sort(key=lambda x: (x.startswith('Ours'), x != 'Ours'))
    if 'Ours' in files:
        files.remove('Ours')
        files.append('Ours')

    # Create a 3x4 subplot layout
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 9))

    # Iterate over all subplots and image files
    for ax, image_file in zip(axes.flatten(), files):
        # Load and display image
        img = plt.imread(os.path.join(file_dir, image_file))
        ax.imshow(img)
        ax.axis('off')  # Turn off axes

    # Adjust subplot spacing and layout
    plt.tight_layout()

    # plt.show()
    save_path = os.path.join(r'.\Figure\3-2.png')

    plt.savefig(save_path)


file_path = r'.\new_exps\comparison'

# draw_f1_curve(file_path)
# draw_R_curve(file_path)
# draw_P_curve(file_path)
# plot_pr_curve(file_path)

# confusion_matrix(file_path)

file_path = r'.\new_exps\confusion-matrix\comparison'
final_confusion(file_path)
