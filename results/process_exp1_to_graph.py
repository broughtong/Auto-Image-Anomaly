import numpy as np
import sys
import matplotlib.pyplot as plt


file_out = sys.argv[1]

file_in1 = "exp1_results.txt"
file_in2 = "exp1_results_neg.txt"


def process_file(filename):
    with open(filename, "r") as f:
        data = f.readlines()
        
    num_of_seeds = len(list(filter(lambda x: x.startswith("#"), data)))

    # remove all nondata lines (like seed or list of classes to add)
    data = filter(lambda x: x.startswith("==") and "%" in x, data)

    # sort data according to the ae order
    data = sorted(data)

    # list all different orders of AEs
    orders = list({int(i.split("=")[2]) for i in data})
    to_idx = dict(zip(orders, range(len(orders))))

    #convert data to matrix - using the sorted nature to determine seed_number
    data_matrix = np.zeros((num_of_seeds, len(orders)))
    last_order = -1
    seed = 0
    imgs = np.zeros_like(data_matrix)
    for line in data:
        s = line.split("=")
        order = int(s[2])
        im = int(s[-1].split(" ")[-2])
        acc = float(s[-1].split("%")[0])/100
        if order != last_order:
            seed = 0
            last_order = order
        imgs[seed, to_idx[order]] = im
        data_matrix[seed, to_idx[order]] = acc
        seed += 1

    #make plottable and remove zeroed elements
    processed = [data_matrix[data_matrix[:, i] != 0, i] for i in range(len(orders))]
    return orders, processed, np.mean(imgs, axis=0).astype(np.int32)

orders_pos, processed_pos, imgs_pos = process_file(file_in1)
orders_neg, processed_neg, imgs_neg = process_file(file_in2)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

ax[1].boxplot(processed_pos, showmeans=False, showfliers=False, meanline=True)
ax[1].set_title('Positive anomalies')
ax[1].set_ylim(0.0, 1.0)
ax[1].yaxis.grid(True)
ax[1].set_xticks([x + 1 for x in range(len(processed_pos))])
#ax1_top = ax[1].twiny()
#ax1_top.tick_params(axis="x", labeltop=True)
ax[1].set_xticklabels(orders_pos)
#ax1_top.set_xticklabels(imgs_pos)

ax[0].boxplot(processed_neg, showmeans=False, showfliers=False, meanline=True)
ax[0].set_title('Negative anomalies')
ax[0].set_ylim(0.0, 1.0)
ax[0].yaxis.grid(True)
ax[0].set_xticks([x + 1 for x in range(len(processed_neg))])
#ax0_top = ax[0].twiny()
#ax0_top.tick_params(labeltop=True, rotation=60)
ax[0].set_xticklabels(orders_neg)
#ax0_top.set_xticklabels(imgs_neg)
#ax0_top.set_xticks([x + 1 for x in range(len(imgs_neg))])

fig.suptitle("Anomaly detection accuracy")
#fig.subplots_adjust(top=.8)

#ax[0].violinplot(processed, showmeans=False, showmedians=True)
#ax[0].yaxis.grid(True)
#ax[0].set_xticks([x + 1 for x in range(len(orders))])
#ax[0].set_xticklabels(orders)
#ax[0].set_title('Violin plot')
plt.savefig(file_out)






