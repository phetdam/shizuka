# tests shizuka.model_selection.resampled_cv. comparison of classification
# performance using logistic regression of different resampling methods versus
# training with cross-validation only. uses shizuka.plotting to produces plots.
# ./resampling_test_config.csv provides configuration information.
#
# Changelog:
#
# 01-28-2020
#
# modified _input_handler to not print extra newline with help blurb.
#
# 01-21-2020
#
# modified to retrieve best estimator from shizukaBaseCV instance attribute.
#
# 01-19-2020
#
# initial creation. more work to be done since resampled_cv return signature
# will be changed to perhaps return an object, not a single dict.

# ignore warnings; sklearn has several FutureWarnings
from warnings import simplefilter
simplefilter(action = "ignore", category = FutureWarning)

from os.path import dirname, exists
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
import sys
# note that shizuka package itself is located in ../
if "../" not in sys.path: sys.path.insert(0, "../")
from shizuka.plotting import multiclass_stats, coef_plot
from shizuka.model_selection import resampled_cv
from textwrap import wrap

_PROGNAME = "resampling_test"
_HELP_STR = """Usage: {0} config_file.csv [ output_file ]\n
test shizuka.model_selection.resampled_cv function. format of config file is:

X_train    X training data file, data shape (n_train, n_features)
X_test     X test data file, data shape (n_test, n_features)
y_train    y training data file, data shape (n_train, 1)
y_test     y test data file, data shape (n_test, 1)
cv         number of cross-validation folds
resampler  resampler name (must be a resampled_cv builtin) or None
figure     output file to write multiclass_stats output to

{0} will print out runtime/fitting statistics and produce plots
for each case by calling shizuka.plotting.multiclass_stats. if output_file is
specified, the output will be redirected to the file instead.
""".format(_PROGNAME)
__doc__ = _HELP_STR

def _input_handler():
    """
    handle input. depending on the values of sys.argv, may either print the help
    blurb or silently do nothing or print a warning. returns tuple exit_status,
    config_file or None, out_file or None. the None is returned if one or more
    files cannot be found; they are the cue for the main to exit. note that if
    the config file exists, but the outfile was specified yet does not exist,
    then both file names will returned as None. if the config file does not exit
    but the outfile exists, then both file names also returned as None.
    """
    # error code, potential file name of config file, output file
    ec, cfn, ofn = 0, None, None
    # no args, print warning
    if len(sys.argv) == 1:
        print("{0}: no arguments. try `{0} --help` for usage".format(_PROGNAME))
        ec = 0
    # only one or two arguments (hopefully .csv file or --help)
    elif (len(sys.argv) == 2) or (len(sys.argv) == 3):
        # if help, print help
        if sys.argv[1] == "--help":
            print(_HELP_STR, end = "")
            ec = 0
        # check if config file exists and if .csv
        elif exists(sys.argv[1]):
            cfn = sys.argv[1]
            # if filename is shorter than .csv extension, not a .csv file
            if (len(cfn) < 4) or (cfn[-4:] != ".csv"):
                print("{0}: error: config file must be .csv file"
                      "".format(_PROGNAME), file = sys.stderr)
                ec, cfn = 1, None
            # else it is .csv file, so return cfn outside if statement
        # else doesn't exist
        else:
            print("{0}: error: cannot find file {1}"
                  "".format(_PROGNAME, sys.argv[1]), file = sys.stderr)
            ec = 2
        # if len(sys.argv) == 3, also check if outfile dir exists
        if len(sys.argv) == 3:
            # outfile and outfile dir
            ofn = sys.argv[2]
            ofn_dir = dirname(ofn)
            # if empty string, set to "."
            ofn_dir = "." if ofn_dir == "" else ofn_dir
            if exists(ofn_dir) == False:
                print("{0}: error: output file directory {1} not found"
                      "".format(_PROGNAME, ofn_dir), file = sys.stderr)
                ec, ofn = 2, None
            # else continue
    # else too many arguments
    else:
        print("{0}: too many arguments. type `{0} --help` for usage"
              "".format(_PROGNAME), file = sys.stderr)
        ec = 1
    # return both file names if successful (ofn may be None)
    return ec, cfn, ofn
        

def _check_ios(df):
    """
    check that the X_train, X_test, y_train, and y_test files exist for each
    case, and that the directories for the output .png files exist. prints
    errors if any of them are missing; returns exit code

    parameters:

    df    configuration DataFrame
    """
    if df is None:
        print("{0}: error: None DataFrame received".format(_PROGNAME),
              file = stderr)
        return 1
    # True if missing any files, False if ok
    missing = False
    for i in df.index:
        # if missing any, print error and set missing to True
        if exists(df.loc[i, "X_train"]) == False:
            print("{0}: cannot find case {1} X_train file {2}"
                  "".format(_PROGNAME, i + 1, df.loc[i, "X_train"]),
                  file = sys.stderr)
            missing = True
        elif exists(df.loc[i, "X_test"]) == False:
            print("{0}: cannot find case {1} X_test file {2}"
                  "".format(_PROGNAME, i + 1, df.loc[i, "X_test"]),
                  file = sys.stderr)
            missing = True
        elif exists(df.loc[i, "y_train"]) == False:
            print("{0}: cannot find case {1} y_train file {2}"
                  "".format(_PROGNAME, i + 1, df.loc[i, "y_train"]),
                  file = sys.stderr)
            missing = True
        elif exists(df.loc[i, "y_test"]) == False:
            print("{0}: cannot find case {1} y_test file {2}"
                  "".format(_PROGNAME, i + 1, df.loc[i, "y_test"]),
                  file = sys.stderr)
            missing = True
        # get directory of figure filename
        fig_dir = dirname(df.loc[i, "figure"])
        if exists(fig_dir) == False:
            print("{0}: cannot find case {1} figure output directory {2}"
                  "".format(_PROGNAME, i, fig_dir), file = sys.stderr)
            missing = True
    # if missing is True, return 1 else return 0
    if missing == True: return 1
    return 0
            
# main
if __name__ == "__main__":
    # run input handler; if config file name cfn is None exit with exit code
    ec, cfn, ofn = _input_handler()
    if cfn is None: quit(ec)
    # if ofn is None, set it to sys.stdout, else open file descriptor
    if ofn is None: ofn = sys.stdout
    else: ofn = open(ofn, "w")
    # get file name of config file and read in config info
    df = read_csv(cfn)
    # check that all the input files exist, that for each case the input files
    # are of appropriate dimension, and that the output file directories exist.
    # if exit code ec > 0, then quit
    ec = _check_ios(df)
    if ec > 0: quit(ec)
    # for each case then fit a logistic regression model and compute/print
    # necessary statistics, while we display to stdout later
    X_train, X_test, y_train, y_test = None, None, None, None
    for i in df.index:
        # uses multinomial loss by default
        lrc = LogisticRegression(penalty = "l2", solver = "lbfgs")
        # read in train and test data files, but be smart: if this file has been
        # read in the previous iteration, don't read it again
        if (i > 0) and \
           df.loc[i, "X_train":"cv"].equals(df.loc[i - 1, "X_train":"cv"]):
            pass
        else:
            X_train = read_csv(df.loc[i, "X_train"])
            X_test = read_csv(df.loc[i, "X_test"])
            y_train = read_csv(df.loc[i, "y_train"])
            y_test = read_csv(df.loc[i, "y_test"])
        # get cv; check if int
        cv = df.loc[i, "cv"]
        try: cv = int(cv)
        except ValueError:
            print("{0}: error: invalid cv value for case {1}"
                  "".format(_PROGNAME, i + 1), file = sys.stderr)
            quit(1)
        # get resampling method; if "None" set to None
        rsm = df.loc[i, "resampler"]
        rsm = None if (rsm == "None") else rsm
        # get results (use builtin scorer); use same splits + single thread
        cv_results = resampled_cv(lrc, X_train, y_train, resampler = rsm,
                                  random_state = 7, cv = cv)
        # either print cv_results or write to file
        print("result for case {0}:\n{1}".format(i + 1, cv_results), file = ofn)
        # get best estimator
        lrc = cv_results.best_estimator
        # write multiclass_stats image to figure file + print message to stdout
        fig_file = df.loc[i, "figure"]
        _ = multiclass_stats(lrc, X_test, y_test, outfile = fig_file)
        print("{0}: case {1} completed. results figure written to {2}"
              "".format(_PROGNAME, i + 1, fig_file))
    # if ofn is not None, close the file
    if ofn is not None: ofn.close()
