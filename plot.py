import matplotlib.pyplot as plt
import csv

psnr = []
ssim = []
with open('SRCNN_stats','r') as file:

    reader = csv.reader(file, skipinitialspace = True)
    for row in reader:
        if not row:
            continue
        psnr.append(float(row[0]))
        ssim.append(float(row[1]))

time = range(len(psnr))  # create time axis

fig, y1 = plt.subplots()
plt.title("Training SRCNN")

y1.plot(psnr, label='PSNR', color='red')
y1.set_xlabel('Frame')
y1.set_ylabel('PSNR')
y1.tick_params(axis='y', labelcolor='red')

# create a secondary axis with a different scale for the second axis
y2 = y1.twinx()
y2.plot(ssim, label='SSIM', color='blue')
y2.set_ylabel('SSIM')
y2.tick_params(axis='y', labelcolor='blue')

# add a legend for both axes
lines, labels = y1.get_legend_handles_labels()
lines2, labels2 = y2.get_legend_handles_labels()
y2.legend(lines + lines2, labels + labels2, loc='best')
plt.show()