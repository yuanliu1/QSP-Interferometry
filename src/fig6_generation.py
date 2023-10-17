import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import (MultipleLocator, FixedLocator, AutoMinorLocator, NullFormatter, ScalarFormatter, FormatStrFormatter)
from scipy import stats

fig, ax = plt.subplots()

degree_list = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17])
losses_list = np.array([0.18169011175632463, 0.12118502410867324, 0.11015173411009575, 0.07892235080803331, 0.0789223788942132, 0.06316587504160327, 0.06148306019227054, 0.05032230153388858, 0.05032235157584255, 0.04314158580476358, 0.042605961286560814, 0.03907880951786245, 0.03284796288037957, 0.03160077916073272])


def log_inverse(x, prefactor):
    return prefactor * np.log2(x) / x

from lmfit import Model

gmodel = Model(log_inverse)
print(f'parameter names: {gmodel.param_names}')
print(f'independent variables: {gmodel.independent_vars}')

params = gmodel.make_params(prefactor = 0.2)

x_eval = np.linspace(1, 20, 1001)
y_eval = gmodel.eval(params, x=x_eval)

result = gmodel.fit(losses_list, params, x=degree_list)

print(result.fit_report())

loglog_fit = np.polyfit(np.log(degree_list), np.log(losses_list), 1)
loglog_slope = loglog_fit[0]
loglog_intercept = loglog_fit[1]
xline = np.linspace(1.8, 18, 1000)
print("Slope:", loglog_slope)
print("Intercept:", loglog_intercept)

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams.update({'font.size': 18})

ax.loglog(degree_list, losses_list, "k.")
ax.loglog(xline, log_inverse(xline, result.params['prefactor'].value), "g--", label = "Theoretical")
ax.plot(xline, np.e ** loglog_intercept * xline ** loglog_slope, "r--", label = "Best-fit Power Law")
plt.xlabel(r'$ d $')
plt.ylabel(r'$ p_{err}$')

res = stats.linregress(np.log(degree_list), np.log(losses_list))
print("Slope and Error: " + str(res.slope) + " +- " + str(res.stderr))


# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '.0f' formatting but don't label
# minor ticks.  The string is used directly, the `StrMethodFormatter` is
# created automatically.
# ax.yaxis.set_major_locator(MultipleLocator(20))
# ax.yaxis.set_major_formatter('{x:.0f}')

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_major_locator(MultipleLocator(0.04))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_locator(FixedLocator([2, 3, 4, 5, 6, 7, 8, 12, 16, 20]))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%.0f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# ax.ticklabel_format(axis = 'x', style = 'plain')
ax.legend()

fig = plt.gcf()
fig.savefig("20231017_loss_qsp_w_theory.png", dpi = 300, bbox_inches = 'tight')
plt.show()
