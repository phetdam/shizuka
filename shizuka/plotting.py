# contains some methods for producing plots of statistics for scikit-learn
# classifiers for multi-class classification problems.
#
# Changelog:
#
# 01-11-2020
#
# removed cbar parameter since the color bar really is unnecessary. updated
# docstring for multiclass_stats (finally), and more or less finished it. even
# added usage examples to the docstring for multiclass_stats. changed plot
# legends for multiclass case in multiclass_stats to include only the class
# labels to save space. labels used to also be prefixed with "class_". removed
# the multiclass keyword arg from multiclass_stats since we can infer the type
# of classification problem from the classes_ attribute of the classifier.
#
# 01-10-2020
#
# changed module name to shizuka.plotting and moved to new directory. this
# used to be called multiclass_stats.py, and was a part of tomodachi_proj.
#
# 01-09-2020
#
# added customizable plot style; turns out you need to call axes_style or
# set_style before you create your figure; the style is not retroactive.
#
# 01-08-2019
#
# worked on multiclass_stats, adding the option to plot a normalized (by row)
# confusion matrix, added an optional parameter to pass color maps and colors
# to the function, added more input checking and ability to infer class labels
# from y_test as well as plot these labels on the confusion matrix, fixed
# computation of macro average AUC to work in multiclass case, added code to
# plot the one vs. rest ROC curves. option to control plot grid style using
# seaborn's setting functions does not work properly (todo). added function
# _get_cmap_colors to get colors from matplotlib color maps that are not too
# contrasting with each other + has contrast-controlling threshold. worked on
# editing docstrings for both the module and multiclass_stats function (wip).
# also added macro AUC entry to stats_dict, and changed original AUC entry to
# also be a list in the case of a multiclass scenario. also finally figured
# out how to make the plot areas square: turns out it was a misinterpretation
# of the matplotlib.pyplot.Axes.axis function documentation on my part.
#
# 01-07-2019
#
# made some changes to the module docstring and made substantial changes to the
# multiclass_stats function; the ability to produce bar plots of feature
# importances or model coefficients has been moved to plot_feature_importances.
# rearranged imports in alphabetical order (lmao). working on getting the
# multiclass_stats function to actually work for the multiclass case, not just
# the binary classification use case. to that end, added extra statistics to
# the stats_dict dictionary, such as macro + micro average precision + recall.
#
# 01-06-2020
#
# happy new year! made some tweaks to multiclass_stats since i use this one
# function so much for classification problems; considering making an official
# python package. changed docstring location and format. added number of classes
# to stats_dict for more ease of use. messed around with different ways of
# formatting the graph aspects so that they look nice; didn't get too far.
#
# 12-12-2019
#
# according to the file history, i made this change 12-04 but never wrote it
# down in the change log nor pushed it to the repository. corrected unbound
# local error that one might get due to referencing of the wrong color palette
# variable when calculating feature importances.
#
# 11-30-2019
#
# modified initial argument type checking since there is erroneous fall-through.
# changed to raise appropriate exceptions for each instance.
#
# 11-27-2019
#
# added short main to warn if user tries to run module as script.
#
# 11-26-2019
#
# added proper support for the coef_ attribute, for both the two-class and the
# one-vs-all multi-class classification coefficient schemes. also added option
# to change the color palette being used for the coefficient graphs/feature
# importance graph for the aesthetic. wish i knew of the ravel() method earlier;
# it simplifies the nested axes plotting problem with multiple rows by simply
# flattening into a 1d array. updated docstring to reflect new changes, and
# also corrected some minor docstring typos; added example to module docstring.
#
# note: i did not actually test the multi-class case; i instead modified the
# single class case to have extra plots and then manually plotted a few more
# coefficient graphs, so performance for a real multi-class classifier that
# returns coef_ with shape (n_classes, n_features) is not guaranteed. but my
# tests imply that it should work fine.
#
# 11-25-2019
#
# started work on adding proper support for the coef_ attribute for both the
# two-class and the multi-class classification cases. updated docstring.
#
# 11-22-2019
#
# initial creation. for some reason the ROC curve produced was not exactly
# matching the ROC curve produced by manual line-by-line plotting, but then i
# realized it was an error in my manual code. hence why having this wrapper
# makes repeated plotting a lot more convenient. i also chose the default sizes
# for the plots to be the maximum width to display well in git without having
# to scroll to the right.

_MODULE_NAME = "shizuka.plotting"

__doc__ = """
a module for quickly compiling and plotting statistics commonly used in multi-
class classification problems, with the goal of simplifying visualizations.

contains methods for producing various statistical  plots for fitted sklearn
or sklearn-compatible classifiers applied to multiclass classification problems.

IMPORTANT: matplotlib<=3.1.0 recommended as 3.1.1 messes up the seaborn heatmap
           annotations. i wrote this module with matplotlib==3.1.0; not sure if
           3.1.2. has fixed the heatmap issue. also, the multiclass_stats
           function requires version >=0.22 of sklearn, as the normalize kwarg
           for sklearn.metrics.confusion_matrix was only recently added in
           version 0.22 (not present in <=0.21).
"""

# takes any matplotlib color map and returns a ListedColormap
from ._utils import _get_cmap_colors
from matplotlib.pyplot import barh, step, subplots
from numpy import ndarray, ravel
from pandas import DataFrame, get_dummies
from seaborn import axes_style, heatmap, lineplot
from sklearn.base import ClassifierMixin, RegressorMixin
# metrics
from sklearn.metrics import (confusion_matrix, precision_score,
                             precision_recall_curve, recall_score,
                             roc_auc_score, roc_curve)
from sys import stderr

def multiclass_stats(mce, X_test, y_test, norm_true = True, figsize = "auto",
                     model_name = "auto", best_model = False,
                     style = "darkgrid", cmaps = "auto", no_return = False,
                     outfile = None):
    """
    produces multiple useful plots for evaluating a multiclass classifier
    implemented in sklearn and returns useful statistics. returns a matplotlib
    Figure with all the plots, a confusion matrix, and the dictionary:

    {"mc_rates": [...], "n_classes": n, "accuracy": a, "precision": b or [...],
     "macro_precision": None or c, "micro_precision": None or d,
     "auc": e or [...], "macro_auc": None or f, "recall": g or [...], 
     "macro_recall": None or h, "micro_recall": None or i}

    for the entries that values noted as being "x or y", the x is the value
    taken in the binary classification case, and the y is the value taken in the
    multiclass classification case, which is inferred from the estimator's
    classes_ attribute, which contains all the unique class labels.

    the function produces a confusion matrix using seaborn's heatmap, a ROC
    curve plot, and a precision-recall curve plot. in the multiclass case, the
    ROC curve plot and precision-recall curve plots will contain the relevant
    curves for each class, which are computed in a one vs. rest fashion. the
    format of the subplot titles can be changed by the model_name and best_model
    parameters and is given by, in order of plots from left to right

    "confusion matrix for [best if best_model is True else blank] [model_name if
    model_name not 'auto' else object name]"

    "ROC curve for [best if best_model is True else blank] [model_name if
    model_name not 'auto' else object name]"

    "PrRc curve for [best if best_model is True else blank] [model_name if
    model_name not 'auto; else object name]"

    notes: in the multiclass case, the behavior of the metrics and the plots
           changes. for example, the ROC curve and precision-recall curve plots
           will not plot a single line, but will each plot n_classes curves in a
           one vs. rest scheme. the AUC, precision, and recall values in the
           returned dict of statistics will be a list of one vs. rest statistics
           in sorted class label order, and the macro and micro average entries
           will each be a float instead of None.

    example usage:

    suppose we are given a fitted classifier cf, test data X_test, y_test, and
    want to indicate that cf is the best model out of a few other models, with 
    the name my_best_model_20. we want to save the image to ./cf_stats.png. we
    also want to keep the returned figure, confusion matrix, and dictionary of
    metrics such as AUC, ROC, precision, and recall scores. furthermore, suppose
    that this is a multiclass classification problem.

    therefore, we could use the following function call:

    from shizuka.plotting import multiclass_stats
    fig, cmat, stats_dict = multiclass_stats(cf, X_test, y_test,
                                             best_model = True,
                                             model_name = "my_best_model_20",
                                             outfile = "./cf_stats.png")

    suppose we also wanted to print out some metrics like our ovr precision
    scores, macro and micro precision scores, and ovr AUC scores. we can write

    print("best test macro precision:\\t{0:.5f}\\nbest test micro precision:"
          "\\t{1:.5f}\\nbest test ovr AUC:\\t\\t{2}"
          "".format(stats_dict["macro_precision"],
                    stats_dict["micro_precision"],
                    tuple(map(lambda x: round(x, 5), stats_dict["auc"]))))
    print("best test ovr precision:\\t{0}"
          "".format(tuple(map(lambda x: round(x, 5), stats_dict["precision"]))))

    parameters:

    mce          fitted classifier inheriting from sklearn.base.ClassifierMixin
    X_test       pandas DataFrame test feature matrix
    y_test       test response vector, either one-column DataFrame or ndarray/1d
                 iterable. DataFrame/ndarray recommended.

                 note: mce should already be fit on X_train, y_train data

    norm_true    optional, default True. normalize confusion matrix values over
                 the true class labels (i.e. over the rows), which means for a
                 row i, any cell j != i in row i reports the misclassification
                 rate of class i as class j. set to False to report raw values.

                 note: in the multiclass case, with class imbalance, reporting
                       raw values will introduce scaling issues, so the
                       distribution of colors from the color map will likely
                       result in many cells looking similar in color.

    figsize      optional, default "auto", which is (12, 4) for the binary
                 classification case and is (15, 5) for the multiclass case.
                 tight_layout() and square axes gives square plots.
    model_name   optional, default "auto" gives the class name of mce. changes
                 titles of the confusion matrix, plot of the ROC curve (or
                 curves), and plot of the precision-recall curve (or curves).
    best_model   optional, default False. whether or not to include "best"
                 before the name of the model in the subplot titles.
    style        optional string, default "darkgrid". use it to set the plot
                 area; recognized values are "darkgrid", "whitegrid", "dark",
                 "white", and "ticks", the values recognized by seaborn.set.
    cmaps        optional, default "auto". if user-defined, gives color maps for
                 heatmap (confusion matrix), ROC curve(s), and precision-recall
                 curve(s) respectively. must be a tuple of length 3. in the
                 multiclass case, all elements are treated as color maps, and
                 must be valid matplotlib color map strings. in the binary
                 classification case, cmaps[0] is treated as a color map, and
                 cmaps[1], cmaps[2] are treated as single color strings, which
                 must either be known matplotlib colors or hex color strings of
                 the format "#rrggbb". the multiclass defaults are ("Blues",
                 "Dark2", "tab10"), while the binary class color defaults are
                 ("Blues", "coral", "#DE6FA1"). #DE6FA1 is liseran purple.
    no_return    optional, default False. if True, then returns None instead of
                 the figure, confusion matrix, and dictionary tuple. useful when
                 displaying inline plots using matplotlib in jupyter notebooks.
    outfile      optional, default None. if a string, the method will attempt to
                 save to the figure into that file.
    """
    # save the name of the function for convenience
    fname_ = multiclass_stats.__name__
    # check that the estimator mce is a classifier. if not, print error
    if isinstance(mce, ClassifierMixin) == False:
        raise TypeError("{0}: must pass in classifier inheriting from "
                        "sklearn.base.ClassifierMixin".format(fname_))
    # check the length of X_test and y_test are the same
    if len(X_test) != len(y_test):
        raise ValueError("{0}: X_test and y_test must have same number of "
                         "observations".format(fname_))
    # check type of X_test and y_test
    if not isinstance(X_test, DataFrame):
        raise TypeError("{0}: X_test should be a pandas DataFrame"
                        "".format(fname_))
    # be more careful when checking type of y_test
    if hasattr(y_test, "__iter__") and (not isinstance(y_test, str)):
        # if y_test is a DataFrame, check that there is only one column
        if isinstance(y_test, DataFrame):
            if len(y_test.columns) > 1:
                raise ValueError("{0}: y_test ({1}) must have one column only"
                                 "".format(fname_, type(y_test)))
        # else if y_test is a numpy array, also check ndim
        elif isinstance(y_test, ndarray):
            if y_test.ndim != 1:
                raise ValueError("{0}: y_test ({1}) must have ndim == 1"
                                 "".format(fname_, type(y_test)))
        # don't allow dictionary
        elif isinstance(y_test, dict):
            raise TypeError("{0}: y_test ({1}) must be a 1d iterable"
                            "".format(fname_))
        # don't allow multidimensionality
        for val in y_test:
            if (not isinstance(val, str)) and hasattr(val, "__iter__"):
                raise TypeError("{0}: y_test must be a 1d iterable"
                                "".format(fname_))
    else:
        raise TypeError("{0}: y_test should be a pandas DataFrame with one "
                        "column or a 1d iterable (ndarray, etc.)"
                        "".format(fname_))
    # get class labels from the .classes_ property of the classifier and number
    # of classes by taking the length of clabs. will be used throughout the
    # rest of the function, and are also needed for color map assignment.
    clabs = mce.classes_
    nclasses = len(clabs)
    # check color maps; if "auto", check if nclasses > 2, and assign colors or
    # color maps. else check that length == 3 and that elements are str.
    if cmaps == "auto":
        if nclasses > 2: cmaps = ("Blues", "Dark2", "tab10")
        else: cmaps = ("Blues", "coral", "#DE6FA1")
    elif hasattr(cmaps, "__iter__") and (not isinstance(cmaps, str)):
        pass
    # else raise TypeError
    else:
        raise TypeError("{0}: cmaps must either be \"auto\" or (str, str, str)"
                        " of valid colors or color maps".format(fname_))
    # if norm_true is True, set to "true", else if False, set to None. if not
    # boolean, raise a TypeError to the user
    if isinstance(norm_true, bool):
        if norm_true == True: norm_true = "true"
        else: norm_true = None
    else: raise TypeError("{0}: error: norm_true must be bool".format(fname_))
    # dictionary of statistics
    stats_dict = {}
    # compute confusion matrix
    cmat = confusion_matrix(y_test, mce.predict(X_test), normalize = norm_true)
    # compute misclassification rates
    mc_rates = [None for _ in range(nclasses)]
    for i in range(nclasses):
        # misclassification rate is 1 - correct / sum of all
        mc_rates[i] = 1 - cmat[i][i] / sum(cmat[i])
    # add entry in stats_dict
    stats_dict["mc_rates"] = mc_rates
    # add number of classes to stats_dict
    stats_dict["n_classes"] = nclasses
    # predict values from X_test and get accuracy to add to stats_dict
    y_test_pred = mce.predict(X_test)
    stats_dict["accuracy"] = mce.score(X_test, y_test)
    # compute precision and add to stats_dict; if nclasses > 2, then the entry
    # for precision is an array of nclasses labels (for one vs. rest precision).
    # "macro_precision" will be the macro average (average of all individual
    # precision statistics) and "micro_precision" will be the micro average
    # (total tp / total tp + total fp). in the two-class case, the
    # "micro_precision" and "macro_precision" keys will be None, and "precision"
    # will only be a single scalar value returned by precision_score.
    if nclasses > 2:
        stats_dict["precision"] = precision_score(y_test, y_test_pred,
                                                  average = None)
        stats_dict["macro_precision"] = precision_score(y_test, y_test_pred,
                                                        average = "macro")
        stats_dict["micro_precision"] = precision_score(y_test, y_test_pred,
                                                        average = "micro")
    else:
        stats_dict["precision"] = precision_score(y_test, y_test_pred)
        stats_dict["macro_precision"] = None
        stats_dict["micro_precision"] = None
    # compute ROC AUC and add to stats dict. in the multiclass case, "auc" will
    # be a list of nclasses labels (computed in one vs. rest fashion), while
    # "macro_auc" will have macro average of all AUC scores. we need to binarize
    # our multiclass predictions so we change shape from (N, 1) to (N, nclasses)
    y_test_bins, y_test_pred_bins = None, None
    if nclasses > 2:
        # wrap y_test as a DataFrame and call get_dummies to one-hot encode for
        # the multiclass case. first need to treat all the entries as a string
        # or else get_dummies will not binarize. column names are "class_k" for
        # each label k in our multiclass problem.
        # if y_test is a DataFrame, use iloc to index
        if isinstance(y_test, DataFrame):
            y_test_bins = get_dummies(DataFrame(map(str, y_test.iloc[:, 0]),
                                                columns = ["class"]))
        # else just apply map and wrap in DataFrame
        else:
            y_test_bins = get_dummies(DataFrame(map(str, y_test),
                                                columns = ["class"]))
        # do the same for y_pred_test, which is a 1d iterable
        y_test_pred_bins = get_dummies(DataFrame(map(str, y_test_pred),
                                                 columns = ["class"]))
        # replace "class_" in each of the columns with empty string
        y_test_bins.columns = tuple(map(lambda x: x.replace("class_", ""),
                                        list(y_test_bins.columns)))
        y_test_pred_bins.columns = tuple(map(lambda x: x.replace("class_", ""),
                                        list(y_test_pred_bins.columns)))
        # calculate one vs. rest AUC scores
        stats_dict["auc"] = roc_auc_score(y_test_bins, y_test_pred_bins,
                                          multi_class = "ovr", average = None)
        # calculate macro average or AUC scores (average = "macro")
        stats_dict["macro_auc"] = roc_auc_score(y_test_bins, y_test_pred_bins,
                                                multi_class = "ovr")
    else:
        stats_dict["auc"] = roc_auc_score(y_test, y_test_pred)
        stats_dict["macro_auc"] = None
    # compute recall and add to stats dict. in the multiclass case, the "recall"
    # key will hold an array of one vs. rest recall scores, while it will be a
    # single float in the binary classification case. "macro_recall" and
    # "micro_recall" will give macro and micro recall scores, and will be None
    # in the binary classification case.
    if nclasses > 2:
        stats_dict["recall"] = recall_score(y_test, y_test_pred, average = None)
        stats_dict["macro_recall"] = recall_score(y_test, y_test_pred,
                                                  average = "macro")
        stats_dict["micro_recall"] = recall_score(y_test, y_test_pred,
                                                  average = "micro")
    else:
        stats_dict["recall"] = recall_score(y_test, y_test_pred)
        stats_dict["macro_recall"] = None
        stats_dict["micro_recall"] = None
    # compute true and false positive rates for the ROC curve. if nclasses > 2,
    # then fpr and tpr will be 2d arrays, where each row i contains the fpr or
    # tpr for the one vs. all ROC curve for class label i.
    fpr, tpr = None, None
    if nclasses > 2:
        # set up fpr and tpr as being length nclasses
        fpr = [None for _ in range(nclasses)]
        tpr = [None for _ in range(nclasses)]
        # for each of the labels in clabs corresponding to a column in
        # y_test_bins and y_test_pred_bins, compute one vs. all fpr and tpr. we
        # do not need to specify positive label since y_test_bins and
        # y_test_pred_bins are both indicator matrices.
        for i in range(nclasses):
            fpr[i], tpr[i], _ = roc_curve(y_test_bins.iloc[:, i],
                                          y_test_pred_bins.iloc[:, i])
    # else just compute as usual for y_test, y_test_pred
    else: fpr, tpr, _ = roc_curve(y_test, y_test_pred)
    # compute precision-recall curves. if nclasses > 2, then prr and rcr
    # (precision and recall rates respectively) will be 2d arrays, where each
    # row i contains the prr or rcr for the ovr precision-recall curve for i.
    prr, rcr = None, None
    if nclasses > 2:
        # set up prr and rcr as being length nclasses
        prr = [None for _ in range(nclasses)]
        rcr = [None for _ in range(nclasses)]
        # for each label in clabs corresponding to a column in y_test_bins and
        # y_test_pred_bins, compute one vs. all prr and rcr.
        for i in range(nclasses):
            prr[i], rcr[i], _ = \
                precision_recall_curve(y_test_bins.iloc[:, i],
                                       y_test_pred_bins.iloc[:, i])
    # else just compute as usual using y_test, y_test_pred
    else: prr, rcr, _ = precision_recall_curve(y_test, y_test_pred)
    ### figure setup ###
    # if figsize is "auto" (default), determine plot size based on whether the
    # problem is a binary classification problem or a multiclass one.
    if figsize == "auto":
        if nclasses > 2: figsize = (15, 5)
        else: figsize = (12, 4)
    # else figsize is user specified; generate subplots with specified style
    with axes_style(style = style):
        fig, axs = subplots(nrows = 1, ncols = 3, figsize = figsize)
    # flatten the axes (for ease of iterating through them)
    axs = ravel(axs)
    # forces all plot areas to be square (finally!)
    for ax in axs: ax.axis("square")
    # set best option
    best_ = ""
    if best_model is True: best_ = "best "
    # set model name; if auto, set to object name
    if model_name == "auto": model_name = str(mce).split("(")[0]
    ### create confusion matrix ###
    # set title of confusion matrix
    axs[0].set_title("confusion matrix for {0}{1}".format(best_, model_name))
    # heatmap, with annotations, and using clabs as axis labels. if normalized,
    # report percentage (no decimal places), else report as decimal value
    if norm_true == "true":
        heatmap(cmat, annot = True, cmap = cmaps[0], cbar = False,
                xticklabels = clabs, yticklabels = clabs, ax = axs[0],
                fmt = ".0%")
    else:
        heatmap(cmat, annot = True, cmap = cmaps[0], cbar = False,
                xticklabels = clabs, yticklabels = clabs, ax = axs[0],
                fmt = "d")
    ### create plot of ROC curves ###
    # set axis labels; same in multiclass and binary case
    axs[1].set_xlabel("false positive rate")
    axs[1].set_ylabel("true positive rate")
    # if we have the multiclass case
    if nclasses > 2:
        axs[1].set_title("ROC curves for {0}{1}".format(best_, model_name))
        # get ListedColormap from the specified color map string
        lcm = _get_cmap_colors(cmaps[1], nclasses, callfn = fname_)
        # for each class label in clabs, plot its respective true positive rate
        # against its false positive rate with lcm.colors[i] as the color
        for i in range(nclasses):
            axs[1].plot(fpr[i], tpr[i], color = lcm.colors[i])
        # set legend from y_test_bins's column names
        axs[1].legend(y_test_bins.columns)
    # else just create one simple line plot with a single color
    else:
        axs[1].set_title("ROC curve for {0}{1}".format(best_, model_name))
        axs[1].plot(fpr, tpr, color = cmaps[1])
    ### create plot of precision-recall curves ###
    # set axis labels
    axs[2].set_xlabel("recall")
    axs[2].set_ylabel("precision")
    # if we have the multiclass case
    if nclasses > 2:
        axs[2].set_title("PrRc curves for {0}{1}".format(best_, model_name))
        # get ListedColormap from the specified color map string
        lcm = _get_cmap_colors(cmaps[2], nclasses, callfn = fname_)
        # for each class label in clabs, plot its respective precision against
        # its recall by indexing color with lcm.colors[i]
        for i in range(nclasses):
            axs[2].plot(rcr[i], prr[i], color = lcm.colors[i])
        # set legend from y_test_bins's column names
        axs[2].legend(y_test_bins.columns)
    # else just create one simple line plot with a single color
    else:
        axs[2].set_title("PrRc curve for {0}{1}".format(best_, model_name))
        axs[2].plot(rcr, prr, color = cmaps[2])
    # add tight layout adjustments
    fig.tight_layout()
    # if out_file is not None, save to outfile
    if outfile is not None: fig.savefig(outfile)
    return fig, cmat, stats_dict

def coef_plot(mce, figsize = "auto", model_name = "auto", best_model = False,
              style = "darkgrid", cmap = "auto", outfile = None):
    """
    given a fitted estimator with a coef_ or feature_importances_, plot model
    coefficients or feature importances (respectively). this method works for
    most standard classifiers in sklearn, including boosting classifiers, tree-
    based models (including random forests). but for stacked, voting, or bagged
    estimators, it is recommended to call coef_plot separately for each
    estimator, as ensemble-type models may have many individual estimators.

    note: in the multiclass case, a one vs. rest scheme is assumed, so each plot
          will also indicate which class is being treated as the positive class.

    parameters:

    est          fitted estimator inheriting from sklearn.base.ClassifierMixin
                 or sklearn.base.RegressorMixin
    figsize      optional, default "auto", which is (12, 4) when multiclass is
                 False and is (15, 5) when multiclass is True. tight_layout()
                 and the use of fixed x and y aspect gives square plots.
    model_name   optional, default "auto" gives the class name of mce. changes
                 titles of the coefficient/feature importances plots.
    best_model   optional, default False. whether or not to include "best"
                 before the name of the model in the subplot titles.
    style        optional string, default "darkgrid". use it to set the plot
                 area; recognized values are "darkgrid", "whitegrid", "dark",
                 "white", and "ticks", the values recognized by seaborn.set.
    cmap         optional, default "Dark2". defines the color map for the plots.
    outfile      optional, default None. if a string, the method will attempt to
                 save to the figure into that file.
    """
    # save the name of the function for convenience
    _fn = coef_plot.__name__
    # check that estimator is either classifier or regressor
    if (isinstance(est, ClassifierMixin) == False) and \
       (isinstance(est, RegressorMixin) == False):
        raise TypeError("{0}: estimator must inherit from sklearn.base."
                        "ClassifierMixin or sklearn.base.RegressorMixin"
                        "".format(_fn))





    # if feature_ws is True, check if either coef_ or feature_importances_ does
    # exist within mce. if an exception is raised, catch it, print error, exit.
    if feature_ws == True:
        # first look for coefficients
        try:
            coefs = getattr(mce, "coef_")
        # else do nothing, as we print errors later
        except: pass
        # then look for feature importances if we have it
        try:
            feature_imps = getattr(mce, "feature_importances_")
        # else do nothing, as we print errors later
        except: pass
        # if errors are verbose, print error for any None values
        if verbose == True:
            if coefs is None:
                print("{0}: error: object\n{1} does not have attribute coef_"
                      "".format(fname_, mce), file = sys.stderr)
            if feature_imps is None:
                print("{0}: error: object\n{1} does not have attribute feature"
                      "_importances_".format(fname_, mce), file = sys.stderr)
        # if both are None, print error and exit
        if (coefs is None) and (feature_imps is None):
            print("{0}: error: object\n{1} does not have attributes coef_ and "
                  "feature_importances_".format(fname_, mce), file = sys.stderr)
            quit(1)
    # number of subplots in the figure; 2 default unless feature_ws is True
    nplots = 2
    # shape of coefs; if coefs is not None, shape is (1, n_features) if there
    # are only two classes. for multiple classes, (n_classes, n_features).
    coefs_shape = None
    # if feature_ws is True, we have to determine three cases: 1. if there is no
    # coefs and only feature_imps is not None, then 3 plots, 1 row. 2. there is
    # coefs so no feature_imps, but only two classes, which means coefs will
    # have shape (1, n_features), so again 3 plots. 3. there is coefs so no
    # feature_imps, but multiple classes, where coefs will have the shape
    # (n_classes, n_features), so n_classes + 2 plots.
    if feature_ws is True:
        # first check that we have coefficients
        if coefs is not None:
            # get the shape
            coefs_shape = (len(coefs), len(coefs[0]))
            # if coefs_shape[0] == 1, then set nplots = 3
            if coefs_shape[0] == 1: nplots = 3
            # else > 1 so set nplots = 2 + coefs_shape[0]
            else: nplots = 2 + coefs_shape[0]
        # else if coefs is None, then feature_imps is not None (we already did
        # error checking in the the previous statement already), so nplots = 3
        elif feature_imps is not None: nplots = 3
    # if figsize is "auto", do 4 inch width + height if nplots == 2, 4.2666
    # inches width and 4 inches height/subplot is nplots == 3, and 4.65 inches
    # width and 4 inches height/subplot if nplots > 3 in order to maintain
    # differing square shapes based on tight_layout() adjustments.
    if figsize == "auto":
        """
        if nplots == 2: figsize = (4 * nplots, 4)
        if nplots == 3: figsize = (4.2666 * nplots, 4)
        # add an extra row if nplots % 3 > 0; i.e. nplots / 3 > nplots // 3 so
        # there is an extra row needed for the remaining 1 or 2 plots
        else: figsize = (4.65 * 3,
                         4 * (nplots // 3 + (1 if nplots % 3 > 0 else 0)))
        """
        if nplots == 2: figsize = (8, 4)
        if nplots == 3: figsize = (16, 4)
    # if feature_ws is True, then also display feature importances/coefficients
    # based on whatever is first found to be true. first get color palette
    # if coefs is not None, then plot coefficients of the model (harder)
    if coefs is not None:
        # will use same color palette, specified by palette argument, with
        # coefs_shape[1] colors, for each of the coefficients graphs
        colors = sns.color_palette(palette = palette, n_colors = coefs_shape[1])
        # for each remaining subplot, index 2 to nplots - 1, make barplots using
        # palette colors for each of the plots, setting title and plotting.
        # note special case if coefs_shape[0] == 1, where only one set of coefs.
        if coefs_shape[0] == 1:
            axs[2].set_title("coefficients, {0}".format(model_name))
            # get BarContainer from barplot routine
            fakes = [i for i in range(len(X_test.columns))]
            #sns.barplot(data = DataFrame([coefs[0]], columns = X_test.columns),
            sns.barplot(data = DataFrame([coefs[0]], columns = fakes),
                        palette = colors, ax = axs[2], orient = "h")
            print(dir(axs[2].xaxis))
            print(axs[2].yaxis.get_majorticklabels()[0].get_window_extent(dpi = 100))
            # align on edge is easier to do by hand
            #barh(coefs[0], align = "edge")
            #for pc in cfbc.patches: pc.width = 0.01
            # adjust heights of the bars
            #lineplot(fpr, tpr, color = "magenta", ax = axs[2]) # this works
        # else there are multiple sets of coefficients, so in the title for each
        # of the barplots, index by the class number
        else:
            for i in range(2, nplots):
                axs[i].set_title("coefficients {0}, {1}".format(i - 2,
                                                                model_name))
                sns.barplot(data = DataFrame([coefs[i - 2]],
                                             columns = X_test.columns),
                            palette = colors, ax = axs[i], orient = "h")
    # else if feature_imps is not None, plot the feature importances
    elif feature_imps is not None:
        axs[2].set_title("feature importances, {0}".format(model_name))
        sns.barplot(data = DataFrame([feature_imps], columns = X_test.columns),
                    palette = palette, ax = axs[2], orient = "h")
    # if both are None, do nothing; feature_ws is probably False
    # adjust figure for tightness
    fig.tight_layout()
    # if outfile is not None, save to outfile
    if outfile is not None: fig.savefig(outfile)
    # if no_return is True, return None
    if no_return == True: return None
    # else return figure, confusion matrix cmat, and statistics in stats_dict
    return fig, cmat, stats_dict

# main
if __name__ == "__main__":
    print("{0}: do not run module as script. refer to docstring for usage."
          "".format(_MODULE_NAME), file = stderr)
