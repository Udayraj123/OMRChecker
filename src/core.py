"""

Designed and Developed by-
Udayraj Deshmukh
https://github.com/Udayraj123

"""

# import sys
from glob import glob
from csv import QUOTE_NONNUMERIC
from time import localtime, strftime, time
from pathlib import Path
from .config import openTemplateWithDefaults, openConfigWithDefaults
from .processors.manager import ProcessorManager
from .template import Template

import src.utils
import src.constants

import imutils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import re
import os


# Load processors
processorManager = ProcessorManager()

# TODO: Move these globals into a class
filesMoved=0
filesNotMoved=0

# TODO(beginner task) :-
# from colorama import init
# init()
# from colorama import Fore, Back, Style

def entry_point(root_dir, curr_dir, args):
    return process_dir(root_dir, curr_dir, args)

# TODO: make this function pure
def process_dir(root_dir, curr_dir, args, template):

    # Update local template (in current recursion stack) 
    local_template_path = curr_dir.joinpath(constants.TEMPLATE_FILENAME)
    if os.path.exists(local_template_path):
        template = Template(local_template_path, processorManager.processors)

    # Look for subdirectories for processing
    subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]

    paths = config.Paths(Path(args['output_dir'], curr_dir.relative_to(root_dir)))

    # look for images in current dir to process
    exts = ('*.png', '*.jpg')       
    omr_files = sorted(
        [f for ext in exts for f in curr_dir.glob(ext)])

    # Exclude images (take union over all preprocessors)
    excluded_files = []
    if(template):
        for pp in template.preprocessors:
            excluded_files.extend(Path(p) for p in pp.exclude_files())

    omr_files = [f for f in omr_files if f not in excluded_files]
    
    if omr_files:
        if not template:
            print(f'Error: Found images, but no template in the directory tree of "{curr_dir}". \nPlace {constants.TEMPLATE_FILENAME} in the directory or specify a template using -t.')
            return
        
        # TODO: get rid of args here 
        args_local = args.copy()
        if("OverrideFlags" in template.options):
            args_local.update(template.options["OverrideFlags"])
        print('\n------------------------------------------------------------------')
        print(f'Processing directory "{curr_dir}" with settings- ')
        print("\tTotal images       : %d" % (len(omr_files)))
        print("\tCropping Enabled   : " + str("croponmarkers" in template.preprocessors))
        print("\tAuto Alignment     : " + str(args_local["autoAlign"]))
        print("\tUsing Template     : " + str(template))
        # Print options
        for pp in template.preprocessors:
            print(f'\tUsing preprocessor "{pp.__class__.__name__:13}({pp})"')

        print('')

        utils.setup_dirs(paths)
        output_set = setup_output(paths, template)
        process_files(omr_files, template, args_local, output_set)

    elif len(subdirs) == 0:
        # Each subdirectory should have images or should be non-leaf
        print(f'Note: No valid images or subfolders found in {curr_dir}. Empty directories not allowed.')

    # recursively process subfolders
    for d in subdirs:
        process_dir(root_dir, d, args, template)


def checkAndMove(error_code, filepath, filepath2):
    # print("Dummy Move:  "+filepath, " --> ",filepath2)
    global filesNotMoved
    filesNotMoved += 1
    return True

    global filesMoved
    if(not os.path.exists(filepath)):
        print('File already moved')
        return False
    if(os.path.exists(filepath2)):
        print('ERROR : Duplicate file at ' + filepath2)
        return False

    print("Moved:  " + filepath, " --> ", filepath2)
    os.rename(filepath, filepath2)
    filesMoved += 1
    return True


def processOMR(template, omrResp):
    # Note: This is a reference function. It is not part of the OMR checker
    # So its implementation is completely subjective to user's requirements.
    csvResp = {}

    # symbol for absent response
    UNMARKED_SYMBOL = ''

    # print("omrResp",omrResp)

    # Multi-column/multi-row questions which need to be concatenated
    for qNo, respKeys in template.concats.items():
        csvResp[qNo] = ''.join([omrResp.get(k, UNMARKED_SYMBOL)
                                for k in respKeys])

    # Single-column/single-row questions
    for qNo in template.singles:
        csvResp[qNo] = omrResp.get(qNo, UNMARKED_SYMBOL)

    # Note: Concatenations and Singles together should be mutually exclusive
    # and should cover all questions in the template(exhaustive)
    # TODO: ^add a warning if omrResp has unused keys remaining
    return csvResp


def report(
        Status,
        streak,
        scheme,
        qNo,
        marked,
        ans,
        prevmarks,
        currmarks,
        marks):
    print(
        '%s \t %s \t\t %s \t %s \t %s \t %s \t %s ' % (qNo,
                                                       Status,
                                                       str(streak),
                                                       '[' + scheme + '] ',
                                                       (str(prevmarks) + ' + ' + str(currmarks) + ' =' + str(marks)),
                                                       str(marked),
                                                       str(ans)))


def setup_output(paths, template):
    ns = argparse.Namespace()
    print("\nChecking Files...")

    # Include current output paths
    ns.paths = paths

    # custom sort: To use integer order in question names instead of
    # alphabetical - avoids q1, q10, q2 and orders them q1, q2, ..., q10
    ns.respCols = sorted(list(template.concats.keys()) + template.singles,
                         key=lambda x: int(x[1:]) if ord(x[1]) in range(48, 58) else 0)
    ns.emptyResp = [''] * len(ns.respCols)
    ns.sheetCols = ['file_id', 'input_path',
                    'output_path', 'score'] + ns.respCols
    ns.OUTPUT_SET = []
    ns.filesObj = {}
    ns.filesMap = {
        "Results": paths.RESULTS_DIR + 'Results_' + timeNowHrs + '.csv',
        "MultiMarked": paths.MANUAL_DIR + 'MultiMarkedFiles_.csv',
        "Errors": paths.MANUAL_DIR + 'ErrorFiles_.csv',
        "BadRollNos": paths.MANUAL_DIR + 'BadRollNoFiles_.csv'
    }

    for fileKey, fileName in ns.filesMap.items():
        if(not os.path.exists(fileName)):
            print("Note: Created new file: %s" % (fileName))
            # moved handling of files to pandas csv writer
            ns.filesObj[fileKey] = fileName
            # Create Header Columns
            pd.DataFrame([ns.sheetCols], dtype=str) \
              .to_csv(ns.filesObj[fileKey], 
                      mode='a', 
                      quoting=QUOTE_NONNUMERIC, 
                      header=False, 
                      index=False)
        else:
            print('Present : appending to %s' % (fileName))
            ns.filesObj[fileKey] = open(fileName, 'a')

    return ns


''' TODO: Refactor into new process flow.
    Currently I have no idea what this does so I left it out'''

def preliminary_check():
    pass
    # filesCounter=0
    # mws, mbs = [],[]
    # # PRELIM_CHECKS for thresholding
    # if(config.PRELIM_CHECKS):
    #     # TODO: add more using unit testing
    #     TEMPLATE = TEMPLATES["H"]
    #     ALL_WHITE = 255 * np.ones((TEMPLATE.dims[1],TEMPLATE.dims[0]), dtype='uint8')
    #     OMRresponseDict,final_marked,MultiMarked,multiroll = readResponse("H",ALL_WHITE,name = "ALL_WHITE", savedir = None, autoAlign=True)
    #     print("ALL_WHITE",OMRresponseDict)
    #     if(OMRresponseDict!={}):
    #         print("Preliminary Checks Failed.")
    #         exit(2)
    #     ALL_BLACK = np.zeros((TEMPLATE.dims[1],TEMPLATE.dims[0]), dtype='uint8')
    #     OMRresponseDict,final_marked,MultiMarked,multiroll = readResponse("H",ALL_BLACK,name = "ALL_BLACK", savedir = None, autoAlign=True)
    #     print("ALL_BLACK",OMRresponseDict)
    #     show("Confirm : All bubbles are black",final_marked,1,1)



# TODO: take a look at 'out.paths'
def process_files(omr_files, template, args, out):
    start_time = int(time())
    global filesNotMoved
    filesCounter = 0
    filesNotMoved = 0

    for filepath in omr_files:
        filesCounter += 1

        filename = filepath.name
        args['current_file'] = filepath

        inOMR = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        print('')
        print(f'({filesCounter}) Opening image: \t{filepath}\tResolution: {inOMR.shape}')

        # TODO: Get rid of saveImgList
        for i in range(config.outputs.save_image_level):
            utils.resetSaveImg(i+1)

        utils.appendSaveImg(1, inOMR)

        # resize to conform to template
        inOMR = utils.resize_util(
            inOMR, config.dimensions.processing_width, config.dimensions.processing_height)

        # run preprocessors in sequence
        for preprocessor in template.preprocessors:
            inOMR = preprocessor.apply_filter(inOMR, args)           
        
        if(inOMR is None):
            # Error OMR case
            newfilepath = out.paths.ERRORS_DIR + filename
            out.OUTPUT_SET.append([filename] + out.emptyResp)
            if(checkAndMove(constants.ERROR_CODES.NO_MARKER_ERR, filepath, newfilepath)):
                err_line = [filename, filepath,
                            newfilepath, "NA"] + out.emptyResp
                pd.DataFrame(err_line, dtype=str) \
                  .T                              \
                  .to_csv(out.filesObj["Errors"],
                          mode='a',
                          quoting=QUOTE_NONNUMERIC,
                          header=False,
                          index=False)
            continue

        if(args["setLayout"]):
            templateLayout = utils.drawTemplateLayout(
                inOMR, template, shifted=False, border=2)
            utils.show("Template Layout", templateLayout, 1, 1)
            continue

        # uniquify
        file_id = str(filename)
        savedir = out.paths.SAVE_MARKED_DIR
        OMRresponseDict, final_marked, MultiMarked, multiroll = \
            utils.readResponse(template, inOMR, name=file_id,
                         savedir=savedir, autoAlign=args["autoAlign"])

        # concatenate roll nos, set unmarked responses, etc
        resp = processOMR(template, OMRresponseDict)
        print("\nRead Response: \t", resp,"\n")
        if(config.outputs.show_image_level >= 1):
            utils.show("Final Marked Bubbles : " + file_id,
                 utils.resize_util_h(final_marked, int(config.dimensions.display_height * 1.3)), 1, 1)

        #This evaluates and returns the score attribute
        # TODO: Automatic scoring
        #score = evaluate(resp, explain_scoring=config.outputs.explain_scoring)
        score = 0
        
        respArray=[]
        for k in out.respCols:
            respArray.append(resp[k])

        out.OUTPUT_SET.append([filename] + respArray)

        # TODO: Add roll number validation here
        if(MultiMarked == 0):
            filesNotMoved += 1
            newfilepath = savedir + file_id
            # Enter into Results sheet-
            results_line = [filename, filepath, newfilepath, score] + respArray
            # Write/Append to results_line file(opened in append mode)
            pd.DataFrame(results_line, dtype=str) \
              .T                                  \
              .to_csv(out.filesObj["Results"],
                      mode='a',
                      quoting=QUOTE_NONNUMERIC,
                      header=False,
                      index=False)
            print("[%d] Graded with score: %.2f" %
                  (filesCounter, score), '\t file_id: ', file_id)
            # print(filesCounter,file_id,resp['Roll'],'score : ',score)
        else:
            # MultiMarked file
            print('[%d] MultiMarked, moving File: %s' %
                  (filesCounter, file_id))
            newfilepath = out.paths.MULTI_MARKED_DIR + filename
            if(checkAndMove(constants.ERROR_CODES.MULTI_BUBBLE_WARN, filepath, newfilepath)):
                mm_line = [filename, filepath, newfilepath, "NA"] + respArray
                pd.DataFrame(mm_line, dtype=str) \
                  .T                             \
                  .to_csv(out.filesObj["MultiMarked"],
                          mode='a',
                          quoting=QUOTE_NONNUMERIC,
                          header=False,
                          index=False)
            # else:
            #     TODO:  Add appropriate record handling here
            #     pass

        # flush after every 20 files for a live view
        # if(filesCounter % 20 == 0 or filesCounter == len(omr_files)):
        #     for fileKey in out.filesMap.keys():
        #         out.filesObj[fileKey].flush()

    timeChecking = round(time() - start_time, 2) if filesCounter else 1
    print('')
    print('Total files moved        : %d ' % (filesMoved))
    print('Total files not moved    : %d ' % (filesNotMoved))
    print('------------------------------')
    print(
        'Total files processed    : %d (%s)' %
        (filesCounter,
         'Sum Tallied!' if filesCounter == (
             filesMoved +
             filesNotMoved) else 'Not Tallying!'))

    if(config.outputs.show_image_level <= 0):
        print(
            '\nFinished Checking %d files in %.1f seconds i.e. ~%.1f minutes.' %
            (filesCounter, timeChecking, timeChecking / 60))
        print('OMR Processing Rate  :\t ~ %.2f seconds/OMR' %
              (timeChecking / filesCounter))
        print('OMR Processing Speed :\t ~ %.2f OMRs/minute' %
              ((filesCounter * 60) / timeChecking))
    else:
        print("\nTotal script time :", timeChecking, "seconds")

    if(config.outputs.show_image_level <= 1):
        # TODO: colorama this
        print(
            "\nTip: To see some awesome visuals, open config.py and increase 'show_image_level'")

    #evaluate_correctness(template, out)

    # Use this data to train as +ve feedback
    # if config.outputs.show_image_level >= 0 and filesCounter > 10:
    #     for x in [thresholdCircles]:#,badThresholds,veryBadPoints, mws, mbs]:
    #         if(x != []):
    #             x = pd.DataFrame(x)
    #             print(x.describe())
    #             plt.plot(range(len(x)), x)
    #             plt.title("Mystery Plot")
    #             plt.show()
    #         else:
    #             print(x)


# Evaluate accuracy based on OMRDataset file generated through moderation
# portal on the same set of images
def evaluate_correctness(template, out):
    # TODO: TEST_FILE WOULD BE RELATIVE TO INPUT SUBDIRECTORY NOW-
    TEST_FILE = 'inputs/OMRDataset.csv'
    if(os.path.exists(TEST_FILE)):
        print("\nStarting evaluation for: " + TEST_FILE)

        TEST_COLS = ['file_id'] + out.respCols
        y_df = pd.read_csv(
            TEST_FILE, dtype=str)[TEST_COLS].replace(
            np.nan, '', regex=True).set_index('file_id')

        if(np.any(y_df.index.duplicated)):
            y_df_filtered = y_df.loc[~y_df.index.duplicated(keep='first')]
            print(
                "WARNING: Found duplicate File-ids in file %s. Removed %d rows from testing data. Rows remaining: %d" %
                (TEST_FILE, y_df.shape[0] - y_df_filtered.shape[0], y_df_filtered.shape[0]))
            y_df = y_df_filtered

        x_df = pd.DataFrame(
            out.OUTPUT_SET,
            dtype=str,
            columns=TEST_COLS).set_index('file_id')
        # print("x_df",x_df.head())
        # print("\ny_df",y_df.head())
        intersection = y_df.index.intersection(x_df.index)

        # Checking if the merge is okay
        if(intersection.size == x_df.index.size):
            y_df = y_df.loc[intersection]
            x_df['TestResult'] = (x_df == y_df).all(axis=1).astype(int)
            print(x_df.head())
            print("\n\t Accuracy on the %s Dataset: %.6f" %
                  (TEST_FILE, (x_df['TestResult'].sum() / x_df.shape[0])))
        else:
            print(
                "\nERROR: Insufficient Testing Data: Have you appended MultiMarked data yet?")
            print("Missing File-ids: ",
                  list(x_df.index.difference(intersection)))


timeNowHrs = strftime("%I%p", localtime())

