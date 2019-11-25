"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""

# In[62]:
import re
import os
import sys
import cv2
import glob
from csv import QUOTE_NONNUMERIC
import argparse
from time import localtime,strftime,time
from random import randint, sample as randomSample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imutils

import config
import utils
from template import Template

# Local globals
filesMoved=0
filesNotMoved=0

# TODO: Sometime later-
# from colorama import init
# init()
# from colorama import Fore, Back, Style

def process_dir(root_dir, subdir, template):
    curr_dir = os.path.join(root_dir, subdir)

    # Look for template in current dir
    template_file = os.path.join(curr_dir, config.TEMPLATE_FILE)
    if os.path.exists(template_file):
        template = Template(template_file)
        print(f'Found template in {curr_dir}')

    # look for images in current dir to process
    paths = config.Paths(os.path.join(args['output_dir'], subdir))   
    exts = ('*.png', '*.jpg')
    omr_files = [f for ext in exts for f in glob.glob(os.path.join(curr_dir, ext))]

    if omr_files:
        if not template:
            print(f'Error: No template file when processing {curr_dir}.')
            print(f'  Place {config.TEMPLATE_FILE} in the directory or specify a template using -t.')
            return

        utils.check_dirs(paths)  
        output_set = setup_output(paths, template)

        print(f'\nProcessing {curr_dir}...')
        print('Additional Modules:')
        print("\tCropping Enabled   : "+str(not args["noCropping"]))
        print("\tMarkers Enabled    : "+str(template.marker))
        print("\tAuto Alignment     : "+str(args["autoAlign"]))
        print("\nTotal images present    : %d" % (len(omr_files)))

        process_files(omr_files, template, output_set)

    # recursively process sub directories
    for file in os.listdir(curr_dir):
        if os.path.isdir(os.path.join(curr_dir, file)):
            process_dir(root_dir, os.path.join(subdir, file), template)
    

def move(error_code, filepath,filepath2):
    global filesNotMoved, filesMoved
    print("Dummy Move:  "+filepath, " --> ",filepath2)

    filesNotMoved += 1
    return True
    # if(error_code!=NO_MARKER_ERR):
    #     print("Error Code: "+str(error_code))
    if(not os.path.exists(filepath)):
        print('File already moved')
        return False
    if(os.path.exists(filepath2)):
        print('ERROR : Duplicate file at '+filepath2)
        return False

    print("Moved:  "+filepath, " --> ",filepath2)
    os.rename(filepath,filepath2)
    filesMoved+=1
    return True


def processOMR(template, omrResp):
    # Note: This is a reference function. It is not part of the OMR checker 
    #       So its implementation is completely subjective to user's requirements.
    
    # Additional Concatenation key
    resp={}
    # symbol for absent response
    UNMARKED = '' # 'X'

    # Multi-Integer Type Qs / RollNo / Name
    for qNo, respKeys in template.concats.items():
        resp[qNo] = ''.join([omrResp.get(k,UNMARKED) for k in respKeys])
    # Normal Questions
    for qNo in template.singles:
        resp[qNo] = omrResp.get(qNo,UNMARKED)
    # Note: Concatenations and Singles together should be mutually exclusive 
    # and should cover all questions in the template(exhaustive)
    # ^TODO add a warning if omrResp has unused keys remaining
    
    return resp

# In[76]:

def report(Status,streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks):
    print('%s \t %s \t\t %s \t %s \t %s \t %s \t %s ' % (qNo,
          Status,str(streak), '['+scheme+'] ',(str(prevmarks)+' + '+str(currmarks)+' ='+str(marks)),str(marked),str(ans)))


def setup_output(paths, template):
    ns = argparse.Namespace()
    print("\nChecking Files...")

    # Include current output paths
    ns.paths = paths

    # Concats + Singles includes : all template keys including RollNo if present
    # sort qNos using integer instead of alphabetically
    ns.respCols = sorted(list(template.concats.keys()) + template.singles, 
                         key=lambda x: int(x[1:]) if ord(x[1]) in range(48,58) else 0)
    ns.emptyResp = ['']*len(ns.respCols)
    ns.sheetCols = ['file_id','input_path','output_path','score']+ns.respCols
    
    ns.OUTPUT_SET = []
    ns.filesObj = {}
    ns.filesMap = {
        "Results":     paths.resultDir+'Results_'+timeNowHrs+'.csv',
        "MultiMarked": paths.manualDir+'MultiMarkedFiles_.csv',
        "Errors":      paths.manualDir+'ErrorFiles_.csv',
        "BadRollNos":  paths.manualDir+'BadRollNoFiles_.csv'
    }
    for fileKey,fileName in ns.filesMap.items():
        if(not os.path.exists(fileName)):
            print("Note: Created new file: %s" % (fileName))
            ns.filesObj[fileKey] = open(fileName, 'a') # still append mode req [THINK!]
            # Create Header Columns
            pd.DataFrame([ns.sheetCols], dtype = str).to_csv(ns.filesObj[fileKey], quoting = QUOTE_NONNUMERIC,header=False, index=False) 
        else:
            print('Present : appending to %s' % (fileName))
            ns.filesObj[fileKey] = open(fileName,'a')

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



def process_files(omr_files, template, out):
    filesCounter = 0
    filesNotMoved = 0
    # HACK: For logging and verbose output. Should refactor into proper logging system
    global curr_filename  # name of currently processing file
    
    for filepath in omr_files:
        filesCounter+=1
        # In windows: all '\' will be replaced by '/'
        filepath = filepath.replace(os.sep,'/')

        # Prefixing a 'r' to use raw string (escape character '\' is taken literally)
        finder = re.search(r'.*/(.*)/(.*)',filepath,re.IGNORECASE)
        if(finder):
            inputFolderName, filename = finder.groups()
        else:
            filename = 'dummyFile'+str(filesCounter)
            print("Error: Filepath not matching to Regex: "+filepath)
            continue
        # set global var for reading
        curr_filename = filename

        inOMR = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        print('')
        print('(%d) Opening image: \t' % (filesCounter), filepath, "\tResolution: ", inOMR.shape)
        # show("inOMR",inOMR,1,1)
        OMRcrop = utils.getROI(inOMR, filename, noCropping=args["noCropping"])

        if template.marker:
            OMRCrop = utils.handle_markers(OMRCrop, template.marker)

        if(OMRcrop is None):
            newfilepath = out.paths.errorsDir+filename
            out.OUTPUT_SET.append([filename] + out.emptyResp)
            if(move(config.NO_MARKER_ERR, filepath, newfilepath)):
                err_line = [filename, filepath, newfilepath, "NA"] + out.emptyResp
                pd.DataFrame(err_line, dtype=str).T.to_csv(out.filesObj["Errors"], quoting = QUOTE_NONNUMERIC,header=False,index=False)
            continue

        if(args["setLayout"]):
            # show("Sample OMR", resize_util_h(OMRcrop,display_height), 0) <-- showimglvl 2 does the job
            templateLayout = utils.drawTemplateLayout(OMRcrop, template, shifted=False, border=2)
            utils.show("Template Layout", templateLayout,1,1)
            continue
        #uniquify
        newfilename = inputFolderName + '_' + filename
        savedir = out.paths.saveMarkedDir
        OMRresponseDict, final_marked, MultiMarked, multiroll = \
            utils.readResponse(template, OMRcrop, name = newfilename, savedir = savedir, autoAlign=args["autoAlign"])

        #convert to ABCD, getRoll,etc
        resp = processOMR(template, OMRresponseDict)
        print("\nRead Response: \t", resp)

        #This evaluates and returns the score attribute
        # TODO: Automatic scoring
        #score = evaluate(resp, explain=explain)
        score = 0
        
        respArray=[]
        for k in out.respCols:
            respArray.append(resp[k])
            
        out.OUTPUT_SET.append([filename]+respArray)
        # if((multiroll or not (resp['Roll'] is not None and len(resp['Roll'])==11))):
        if(MultiMarked == 0):
            filesNotMoved += 1
            newfilepath = savedir+newfilename
            # Enter into Results sheet-
            results_line = [filename,filepath,newfilepath,score]+respArray
            # Write/Append to results_line file(opened in append mode)
            pd.DataFrame(results_line, dtype=str).T.to_csv(out.filesObj["Results"], quoting = QUOTE_NONNUMERIC,header=False,index=False)
            print("[%d] Graded with score: %.2f" % (filesCounter, score), '\t',newfilename)
            # print(filesCounter,newfilename,resp['Roll'],'score : ',score)
        else:
            # MultiMarked file
            print('[%d] MultiMarked, moving File: %s' % (filesCounter, newfilename))
            newfilepath = out.paths.multiMarkedDir+filename
            if(move(config.MULTI_BUBBLE_WARN, filepath, newfilepath)):
                mm_line = [filename,filepath,newfilepath,"NA"]+respArray
                pd.DataFrame(mm_line, dtype=str).T.to_csv(out.filesObj["MultiMarked"], quoting = QUOTE_NONNUMERIC,header=False,index=False)
            # else:
            #     Add appropriate record handling here
            #     pass
                
        #else if: 
        # TODO: Apply validation on columns like roll no to make use of badRollsArray
        
        # flush after every 20 files
        if(filesCounter % 20 == 0):
            for fileKey in out.filesMap.keys():
                out.filesObj[fileKey].flush()

    timeChecking=round(time()-start_time,2) if filesCounter else 1
    print('')
    print('Total files processed : %d ' % (filesCounter))
    print('Total files moved : %d ' % (filesMoved))
    print('Total files not moved (Sum should tally) : %d ' % (filesNotMoved))

    if config.showimglvl <= 0:
        print('\nFinished Checking %d files in %.1f seconds i.e. ~%.1f minutes.' % 
        (filesCounter, timeChecking, timeChecking/60))
        print('OMR Processing Rate :\t  ~%.2f sec/OMR' % (timeChecking/filesCounter))
        print('OMR Processing Speed :\t ~%.2f OMRs/minute' % ((filesCounter*60)/timeChecking))
    else:
        print("\nTotal script time :", timeChecking,"seconds")


    if config.showimglvl <=1:
        # colorama this
        print("\nTip: To see some awesome visuals, open globals.py and increase 'showimglvl'")

    #evaluate_correctness(template, out)

    # Use this data to train as +ve feedback
    if config.showimglvl >= 0 and filesCounter > 10:
        for x in [utils.thresholdCircles]:#,badThresholds,veryBadPoints, mws, mbs]:
            if(x != []):
                x = pd.DataFrame(x)
                print( x.describe() )
                plt.plot(range(len(x)),x)
                plt.title("Mystery Plot")
                plt.show()
            else:
                print(x)




timeNowHrs=strftime("%I%p",localtime())
start_time = int(time())

# construct the argument parse and parse the arguments
argparser = argparse.ArgumentParser()
# https://docs.python.org/3/howto/argparse.html
# store_true: if the option is specified, assign the value True to args.verbose. Not specifying it implies False.
argparser.add_argument("-c", "--noCropping", required=False, dest='noCropping', action='store_true', help="Disable page contour detection - use only when page boundary is visible, e.g. images from mobile camera.")
argparser.add_argument("-a", "--autoAlign", required=False, dest='autoAlign', action='store_true', help="Enable automatic template alignment - use only when the paper was bent slightly when scanning.")
argparser.add_argument("-l", "--setLayout", required=False, dest='setLayout', action='store_true', help="Set up OMR template layout - modify your json file and run again until the template is set.")
argparser.add_argument("-i", "--inputDir", required=False, action='append', dest='input_dir', help="Specify an input directory.")
argparser.add_argument("-o", "--outputDir", default='outputs', required=False, dest='output_dir', help="Specify an output directory.")
argparser.add_argument("-t", "--template", required=False, dest='template', help="Specify a default template if no template file in input directories.")


args, unknown = argparser.parse_known_args()
args = vars(args)
if(len(unknown)>0):
    print("\nError: Unknown arguments:",unknown)
    argparser.print_help()
    exit(1)

if args['input_dir'] == None:
    args['input_dir'] = ['inputs']
    
if args['template']:
    args['template'] = Template(args['template'])

for root in args['input_dir']:
    process_dir(root, '', args['template'])