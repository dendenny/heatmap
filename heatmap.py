#!/grid/common/pkgs/python/v3.9.6/bin/python3.9
import subprocess
from dash import dash, dash_table, dcc, html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import pathlib
import re
from sys import exit
from datetime import datetime
from dash.dependencies import Output, Input#
from plotly.subplots import make_subplots
import random
from dash.exceptions import PreventUpdate
import os.path
import argparse

app = dash.Dash()
# 1 read data from pegasus.log, get df composites and df worker started
# 1a: script to extract composite from pegasus.log file, amd return list of composites
#as dataframe and as a string list
def run_composite_extractor(log_file):
    panicking_pattern = re.compile(".*Worker.*is.panicking")
    pending_pattern = re.compile("\s*Pending:\s\d+")
    rule_pattern = re.compile("\s*\d+:\d+:.*")
    starter_pattern = re.compile(".*Worker\s\d+\sstarted.*")
    non_page_pattern = re.compile('\d+-\d+\s\d+:\d+:\d+.*\sdetected\shigh\snon-page\smemory.*')
    flag_panicking = 0
    flag_high_non_page = 0
    time_worker_stamps_list = []
    worker_started_list = []
    composites_list = []
    high_non_page_time_worker_stamps_list = []
    high_non_page_composites = []
    print('extracting composites...')
    with open(log_file, 'r', encoding= 'latin1') as lines:
        for line in lines:
            if(re.search(starter_pattern, line) is not None):
                started_status = re.search(r'(\d+-\d+\s\d+:\d+:\d+)\s[0-9]+\.[0-9]+s.*Worker\s(\d+)\sstarted.*', line)
                time_stamp =  datetime.strptime(started_status.group(1), "%m-%d %H:%M:%S")
                started_worker = started_status.group(2)
                worker_started_list.append((time_stamp, started_worker))######
            elif (re.search(panicking_pattern, line) is not None):
                match_time_stamp = re.search(r'(\d+-\d+\s\d+:\d+:\d+)\s[0-9]+\.[0-9]+s.*Worker\s(\d+).*', line)
                time_stamp = datetime.strptime(match_time_stamp.group(1), "%m-%d %H:%M:%S")
                time_string = str(time_stamp)
                worker_num = match_time_stamp.group(2)
                time_worker = time_string + ', ' + worker_num#concat time and worker, necessary for K:V pairs
                time_worker_stamps_list.append(time_worker)
                rules = []#initialise new rules found list for this instance
                flag_panicking = 1   
            elif (re.search(rule_pattern, line) is not None) and (flag_panicking == 1):
                match_rule = re.search(r"\s*\d+:(\d+):.*", line)
                rule = match_rule.group(1)
                rules.append(rule) #append each rule to the list
            elif (re.search(pending_pattern, line) is not None) and (flag_panicking == 1):
                flag_panicking = 0 #reset panicking
                composites_list.append(rules) #append the list of found rules to the parent list
            elif (re.search(non_page_pattern, line) is not None):
                match_time_stamp = re.search(r'(\d+-\d+\s\d+:\d+:\d+)\s[0-9]+\.[0-9]+s.*Worker\s(\d+)\sdetected\shigh\snon-page.*total:\s(\d+\.\d+)\sGB.*:\s(\d+\.\d+).*', line)
                time_stamp = datetime.strptime(match_time_stamp.group(1), "%m-%d %H:%M:%S")
                time_string = str(time_stamp)
                worker_num = match_time_stamp.group(2)
                mem_mb = int(float(match_time_stamp.group(3)) * 1000)
                page_peak_mb = int(float(match_time_stamp.group(4)) * 1000)
                non_paged_mem = mem_mb - page_peak_mb
                time_worker = time_string + ', ' + str(non_paged_mem)#concat time and worker, necessary for K:V pairs
                high_non_page_time_worker_stamps_list.append(time_worker)
                non_page_rules = []#initialise new rules found list for this instance
                flag_high_non_page= 1 
            elif (re.search(rule_pattern, line) is not None) and (flag_high_non_page == 1):
                match_rule = re.search(r"\s*\d+:(\d+):.*", line)
                rule = match_rule.group(1)
                non_page_rules.append(rule) #append each rule to the list
            elif (re.search(pending_pattern, line) is not None) and (flag_high_non_page == 1):
                flag_high_non_page = 0 #reset non page
                high_non_page_composites.append(non_page_rules) #append the list of found rules to the parent list
            else:
                pass     
        #merge the list of time/worker stamps and composites
        merged_lists = list(map(lambda x, y: (x,y), time_worker_stamps_list, composites_list ))
        merged_high_non_page_dict = dict(list(map(lambda x, y: (x,y), high_non_page_time_worker_stamps_list, high_non_page_composites)))
        #create a dict from the tuples - K:V = time/worker: rule
        dict_of_merged = dict(merged_lists)
        #create dataframe with time/worker as column headers
        df_composites = pd.DataFrame({key:pd.Series(value) for key, value in dict_of_merged.items()})

        df_worker_started = pd.DataFrame(worker_started_list, columns =['TimeID', 'WorkerID'])
        #method to get a list of unique composites, avoid repitition
        df_high_non_page_composites = pd.DataFrame({key:pd.Series(value) for key, value in merged_high_non_page_dict.items()})
        if df_high_non_page_composites.empty:
            worker_started = str(df_worker_started['TimeID'][0])
            df_high_non_page_composites = pd.DataFrame({worker_started + ', -5000' : ['Nan']})

        def flatten_unique_list(extracted_list):
            set_comps =  {i for b in [[i] if not isinstance(i, list) else flatten_unique_list(i) for i in extracted_list] for i in b}
            return list(set_comps)

        extracted_unique_list = flatten_unique_list(composites_list)
        extracted_unique_high_non_page_list = flatten_unique_list(high_non_page_composites)
        print('composites extracted')
        return df_composites, extracted_unique_list, df_worker_started, df_high_non_page_composites, extracted_unique_high_non_page_list

#get rules from decrypted rules file
def get_rules(rules_input):
    print('Extracting rules...')
    rule_pattern = re.compile("^Composite.*\|\s(.*)")
    df = pd.DataFrame(columns= ['CompositeID', 'EngineName', 'Rules'])
    data_list = []
    try:
        with open(rules_input, 'r') as lines:
                for line in lines:
                    #create dataframe
                    if (re.search(rule_pattern, line) is not None):
                        rules = re.search(r"^Composite\s(\d*)\s(.*)\s\|\s(.*)\s*",line)
                        composite_id = rules.group(1)
                        engine_name = rules.group(2)
                        rules_string = rules.group(3)
                        data_list.append({'CompositeID': composite_id, 'EngineName': engine_name, 'Rules': rules_string})
                    else:
                         pass
                df = pd.DataFrame.from_records(data_list)
        return df
    except FileNotFoundError:
        print('File not accessible')

#2a read data from usr.log, get df usr log panic, finished time
#method for getting the times when panics overlap
def run_usr_log_extractor(usr_log):
    status_pattern = re.compile("^INFO:.*Status:.*Elapsed:.")
    heading_pattern = re.compile("\s*Worker:\s*Active\s*Threads")
    panic_pattern = re.compile("\s*\d+:\s*Panic\s.*")
    data_pattern = re.compile("\s*\d+:\s+\d+")
    warehouse_pattern = re.compile("\s*\d+:\s+Warehouse")
    kill_pattern = re.compile('^WARNING:\sWorker.*killed.*')
    flag_status = 0
    flag_worker = 0
    list_worker_time = []
    list_worker_killed =[]
    list_of_data_dicts = []
    #
    with open(usr_log, 'r') as lines:
        for line in lines:
            if re.search(status_pattern, line) is not None:
                flag_status = 1
                match_status = re.search(r'.*\s\d+-(\d+-\d+\s\d+:\d+:\d+)\s*Elapsed:\s+\d+:\d+:\d+', line)
                time_stamp = datetime.strptime(match_status.group(1), "%m-%d %H:%M:%S")
                finish_time = time_stamp
            elif (re.search(heading_pattern, line) is not None) and (flag_status == 1):
                flag_status = 1
                flag_worker = 1
            elif (re.search(data_pattern, line) or re.search(warehouse_pattern, line) is not None) and (flag_status == 1) and (flag_worker == 1):
                #search for data string pattern, and put digits into groups - each (...) is a group
                match_worker_details = re.search(r'\s*(\d+):\s+.*\s+(\d+)\s+(\d+)\s+(\d+)\s*', line)
                worker_id = match_worker_details.group(1)
                worker_total_memory = int(match_worker_details.group(2))
                worker_active_memory = int(match_worker_details.group(3))
                worker_cpuTime = int(match_worker_details.group(4))
                #key value pairs with data
                dict1 = {"TimeID": time_stamp, 
                        "WorkerID": worker_id,
                         "Total_mem": worker_total_memory, 
                         "Active_mem": worker_active_memory,
                         "CPU_time": worker_cpuTime}
                list_of_data_dicts.append(dict1)#append K:V pairs to list 
            elif re.search(panic_pattern, line) and (flag_status == 1) and (flag_worker == 1):
                #search for data string pattern, and put digits into groups - each (...) is a grou
                match_worker_details = re.search(r'\s*(\d+):\s*Panic.', line)
                worker_id = int(match_worker_details.group(1))
                dict1 = {"TimeID": time_stamp, "WorkerID": worker_id}
                list_worker_time.append(dict1)
            elif re.search(kill_pattern, line) is not None:
                worker = re.search(r'^WARNING:\sWorker\s(\d+)\skilled.*', line)
                worker_id = int(worker.group(1))
                worker_killed = {"TimeID": time_stamp, "WorkerID": worker_id}
                list_worker_killed.append(worker_killed)
            else:
                flag_status = 0
                flag_worker = 0
        
        df_mem = pd.DataFrame(list_of_data_dicts)
        df_usr = pd.DataFrame(list_worker_time, columns= ['TimeID', 'WorkerID'])
        df_killed = pd.DataFrame(list_worker_killed, columns= ['TimeID', 'WorkerID'])
     
        last = df_mem.groupby(by=['WorkerID'], as_index=False).last()
        df_worker_finished = last[['TimeID', 'WorkerID']]
        df_worker_finished['WorkerID'] = df_worker_finished['WorkerID'].astype(np.int64)
        # df_worker_finished = df_worker_finished[df_worker_finished['WorkerID'].isin(df_killed['WorkerID']) == False]
        
        return df_usr, finish_time, df_mem, df_worker_finished, df_killed

#read computer.log files and extract non paged memory data
def heartbeat_extractor(comp_log):
    heartbeat_pattern = re.compile('.*:\sHeartbeat:.*')
    cluster_data_pattern = re.compile('.*Warning:\sLarge\scluster\sData\sTransfer.*')
    computer_id_pattern = re.compile('.*Computer\s\d+\sreceived\stask.*')
    non_paged_mem_list = []
    warning_list = []
    
    with open(comp_log, 'r', encoding= 'latin1') as lines:
        for line in lines:
            if re.search(computer_id_pattern, line) is not None:
                computer_id = re.search(r'.*Computer\s(\d+)\sreceived\stask.*', line)
                computer_id = int(computer_id.group(1))
            elif re.search(heartbeat_pattern, line) is not None:
                heartbeat_status =re.search(r'(\d+-\d+\s\d+:\d+:\d+)\s[0-9]+\.[0-9]s.*Heartbeat:.*mem_mb=(\d+)\spage_peak_mb=(\d*).*', line)
                time_stamp = datetime.strptime(heartbeat_status.group(1), "%m-%d %H:%M:%S")
                mem_mb = int(heartbeat_status.group(2))
                page_peak_mb = int(heartbeat_status.group(3))
                non_paged_mem = mem_mb - page_peak_mb
                heartbeat_dict = {'ComputerID': computer_id, 'TimeID': time_stamp, 'Non_Paged_Mem_MB': non_paged_mem}
                non_paged_mem_list.append(heartbeat_dict)
            elif re.search(cluster_data_pattern, line) is not None:
                cluster_data_status = re.search(r'^(\d+-\d+\s\d+:\d+:\d+)\s[0-9]+\.[0-9]+s.*Warning:\sLarge\scluster\sData\sTransfer\s(.*)', line)
                time_stamp = datetime.strptime(cluster_data_status.group(1), "%m-%d %H:%M:%S")
                warning = cluster_data_status.group(2)
                warning_dict = {'ComputerID': computer_id, 'TimeID': time_stamp, 'WarningID': warning}
                warning_list.append(warning_dict)

        df_non_paged_mem_mb = pd.DataFrame(non_paged_mem_list, columns =['ComputerID', 'TimeID', 'Non_Paged_Mem_MB'])
        df_warning = pd.DataFrame(warning_list, columns =['ComputerID', 'TimeID', 'WarningID'])

        return df_non_paged_mem_mb, df_warning

#get non paged mem data
def get_non_paged_mem_data(directory):
    list_non_paged_mem_dfs = []    
    list_cluster_warning_dfs = [] 
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith('computer') and filename.endswith('.log'):
            df_non_paged_mem, df_warning = heartbeat_extractor(reports_input + '/' + filename)
            list_non_paged_mem_dfs.append(df_non_paged_mem)
            list_cluster_warning_dfs.append(df_warning)
    df_non_paged_mem = pd.concat(list_non_paged_mem_dfs)
    df_cluster_warning = pd.concat(list_cluster_warning_dfs)
    return df_non_paged_mem, df_cluster_warning

#check if a directory exists
def check_dir_exist(input_arg):
    dir_path = input_arg
    return os.path.isdir(dir_path)

#create database
def create_data(input_log_location):        
        input_log_location = input_log_location.rstrip('/')
        #1b: decrypt logs file
        print('decrypting log file...')
        results_dir = input_log_location + '_results'
        output_directory = results_dir + '/pegasus.log'
        db_dir = results_dir + '/db'
        subprocess.run('mkdir -p ' + db_dir, shell=True)
        subprocess.run('/vols/ssvrd_t1_007/meg/commander/workspace/hourly_opt/gcc_v9/latest/output/cdn_hier/lnx86_OPT_gcc_v9/tools.lnx86/bin/pegasus -decrypt '+ input_log_location + '/pegasus.log ' + output_directory, shell= True, )
        decrypted_string = results_dir + '/pegasus.log'
        decrypted_location = pathlib.Path(decrypted_string)

        df_composites, extracted_composites_list, df_worker_started, df_high_non_page_composites, extracted_unique_high_non_page_list = run_composite_extractor(decrypted_location)
        #convert list to string to be passed to script
        string_list = str(extracted_composites_list + extracted_unique_high_non_page_list)
        string_list = string_list.replace(" ", "").replace("'","").replace("[", "").replace("]", "")
        print('Generating Rules Report...')
        subprocess.run('/lan/csv/geomrd1/tools/bin/generate_stats.sh.b71746 --force --run-dir ' + results_dir + '/fanout --log-dir ' + input_log_location + '/ --pegasus /vols/ssvrd_t1_007/meg/commander/workspace/hourly_opt/gcc_v9/latest/output/cdn_hier/lnx86_OPT_gcc_v9/tools.lnx86/bin/pegasus --skip-reports --report-fanout-rules ' + string_list, shell = True)
        print('Rules Report generated')
        #1c get stats.rpt.fanout file
        rules_file_input = results_dir + '/fanout/stats.rpt.fanout'
        print('Extracting rules from stats.rpt.fanout')
        df_rules = get_rules(rules_file_input)
        if df_rules.empty:
            empty_data = pd.DataFrame({'CompositeID': 'no panics', 'EngineName': 'no panics', 'Rules': 'no panics'})
            df_rules.append(empty_data)
            print('No panics, no rules extracted')
        else:
            print('Rules extracted')
        
        print('Extracting from usr.log')
        df_usr, finish_time, df_mem, df_worker_finished, df_worker_killed = run_usr_log_extractor(input_log_location +'/usr.log')
        finish_data = {'TimeID': [finish_time]}
        df_run_finished = pd.DataFrame(finish_data)
        print('usr.log done')
        print('Extracting Non-Paged_memory data')
        df_non_paged_mem, df_cluster_warning = get_non_paged_mem_data(directory)

        #pass all to _reports directory
        df_composites.to_csv(db_dir + '/df_composites.csv', index=False)
        df_worker_started.to_csv(db_dir + '/df_worker_started.csv', index=False)
        df_rules.to_csv(db_dir + '/df_rules.csv', index=False)
        df_usr.to_csv(db_dir + '/df_usr.csv', index=False)
        df_mem.to_csv(db_dir + '/df_mem.csv', index=False)
        df_worker_finished.to_csv(db_dir + '/df_worker_finished.csv', index=False)
        df_worker_killed.to_csv(db_dir + '/df_worker_killed.csv', index=False)
        df_run_finished.to_csv(db_dir + '/df_run_finished.csv', index=False)
        df_high_non_page_composites.to_csv(db_dir + '/df_high_non_page_composites.csv', index=False)
        df_non_paged_mem.to_csv(db_dir + '/df_non_paged_mem.csv', index=False)
        df_cluster_warning.to_csv(db_dir + '/df_cluster_warning.csv', index=False)

#input
parser = argparse.ArgumentParser('Generate heatmap based on pegasus.log and usr.log files')
parser.add_argument('logs_dir', action = 'store', help='Pegasus Logs directory name', type = str)
parser.add_argument('--force', help='Regenerate new results db from input', action="store_true")
args = parser.parse_args()
logs_input = args.logs_dir
reports_input = logs_input + '_results/fanout'
directory = os.fsencode(reports_input)
#read input, force create data if data doent't exist
if args.force and check_dir_exist(logs_input) == True:
    create_data(logs_input)
    input_logs_results_location = logs_input + '_results'
elif check_dir_exist(logs_input) == False:
    print('No directory found, try another input directory')
    exit()
elif check_dir_exist(logs_input + '_results') == True:
    input_logs_results_location = logs_input + '_results'
elif check_dir_exist(logs_input + '_results') == False:
    print('No results directory found, generating results')
    create_data(logs_input)
    input_logs_results_location = logs_input + '_results'
else:
    print('Invalid command')
    exit()

#3 merge pegasusdf and usr_df
def get_xy_for_composites(df_composites, col2_name):
    col_list = df_composites.columns.values.tolist()
    #convert list of strings to list of tuples
    col_list = [tuple(map(str, sub.split(', '))) for sub in col_list]
    df_xy_comp = pd.DataFrame(col_list, columns = ['TimeID', col2_name])
    df_xy_comp['TimeID'] = pd.to_datetime(df_xy_comp['TimeID'], format= "%Y-%m-%d %H:%M:%S")
    df_xy_comp['WorkerID'] = df_xy_comp[col2_name].astype(int)#convert worker obj to int
    return df_xy_comp

#pass reults diectory input location and read data
print('Reading data from ' + input_logs_results_location + '/db')
df_composites = pd.read_csv(input_logs_results_location + '/db/df_composites.csv')
df_high_non_page_composites = pd.read_csv(input_logs_results_location + '/db/df_high_non_page_composites.csv')
df_usr = pd.read_csv(input_logs_results_location + '/db/df_usr.csv')
df_mem = pd.read_csv(input_logs_results_location + '/db/df_mem.csv')
df_worker_started = pd.read_csv(input_logs_results_location + '/db/df_worker_started.csv')
df_worker_finished = pd.read_csv(input_logs_results_location + '/db/df_worker_finished.csv')
df_worker_killed = pd.read_csv(input_logs_results_location + '/db/df_worker_killed.csv')
df_rules = pd.read_csv(input_logs_results_location + '/db/df_rules.csv')
df_run_finished = pd.read_csv(input_logs_results_location + '/db/df_run_finished.csv')
df_non_paged_mem = pd.read_csv(input_logs_results_location + '/db/df_non_paged_mem.csv')
df_cluster_warning = pd.read_csv(input_logs_results_location + '/db/df_cluster_warning.csv')

#convert TimeID column for graphs
df_usr['TimeID'] = pd.to_datetime(df_usr['TimeID'], format= "%Y-%m-%d %H:%M:%S")
df_mem['TimeID'] = pd.to_datetime(df_mem['TimeID'], format= "%Y-%m-%d %H:%M:%S")
df_worker_started['TimeID'] = pd.to_datetime(df_worker_started['TimeID'], format= "%Y-%m-%d %H:%M:%S")
df_worker_finished['TimeID'] = pd.to_datetime(df_worker_finished['TimeID'], format= "%Y-%m-%d %H:%M:%S")
df_worker_killed['TimeID'] = pd.to_datetime(df_worker_killed['TimeID'], format= "%Y-%m-%d %H:%M:%S")
df_run_finished['TimeID'] = pd.to_datetime(df_run_finished['TimeID'], format= "%Y-%m-%d %H:%M:%S")
df_cluster_warning['TimeID'] = pd.to_datetime(df_cluster_warning['TimeID'], format= "%Y-%m-%d %H:%M:%S")
start_time = pd.to_datetime(df_worker_started['TimeID'][0])
finish_time = pd.to_datetime(df_run_finished['TimeID'][0])

#merge databases for heatmap
df_tw_pegasus = get_xy_for_composites(df_composites, 'WorkerID')
df_tw_high_non_page = get_xy_for_composites(df_high_non_page_composites, 'Non_Paged_Mem_MB')
max_worker = df_worker_started['WorkerID'].max()
df_outer_merge = pd.merge(df_tw_pegasus, df_usr, how='outer', on=['TimeID', 'WorkerID']).sort_values('TimeID')

print('tracing graph')
# 5 trace graphs
fig = make_subplots(rows=2, cols=1,shared_xaxes=True, shared_yaxes=False, 
                    specs=[[{"secondary_y": True}], [{"secondary_y": False}]], 
                    row_heights=[1,1], vertical_spacing = 0.01
                        )
# loop to generate active memory line graph traces
workers = np.sort(df_mem.WorkerID.unique())
workers = list(map(int, workers.tolist()))
for worker in workers:
    fig.add_trace(go.Scatter(name = 'Worker ' + str(worker),
                    x=df_mem[df_mem['WorkerID'] == worker]['TimeID'],
                    y = df_mem[df_mem['WorkerID'] == worker]['Active_mem'],
                    mode='lines',
                    hovertemplate = 
                    '<i>Time: <i> %{x}'+
                    '<br><b>Active Memory [MB]:</b>: %{y}<br>',),
                    row=1, col=1,
                    )
    fig.update_xaxes(row=1, col=1, tickformat = '%H:%M:%S')
    fig.update_yaxes(row=1, col=1, title = 'Active Memory [MB]', titlefont=dict(size = 20))   

#panic heatmap traces
if df_outer_merge.empty == False:
    trace1 = go.Histogram2d(
        name = 'Heatmap',
        x=df_outer_merge['TimeID'], 
        y=df_outer_merge['WorkerID'],
        colorscale='turbo',
        xbins= dict(start = start_time, end = finish_time, size = 60000),
        ybins= dict(start = 0.5, end = max_worker + 0.5, size = 1),
        zauto=True,
        showlegend=False,
        showscale  = True,
        colorbar=dict(  title='Panics per min',
                        lenmode="pixels", len=400,
                        xanchor="left", x=0.1,
                        yanchor='bottom', y=-0.2,
                        orientation = 'h'
                        ),
        hovertemplate = 
        '<i>Time: <i> %{x}'+
        '<br><b>Worker:</b>: %{y}<br>', 
        )
    fig.add_trace(trace1,row=2,col=1)
#second trace, giving exact x,y of panic
if df_tw_pegasus.empty == False:
    trace2 = go.Scatter(
        name = 'Panic Points',
        x = df_tw_pegasus['TimeID'], 
        y=df_tw_pegasus['WorkerID'],
        mode = 'markers',
        showlegend=False,
        hovertemplate = 
        '<i>Time: <i> %{x}'+
        '<br><b>Worker:</b>: %{y}<br>', 
        marker=dict(
            symbol='x',
            opacity=0.7,
            color='white',
            size=8,
            line=dict(width=1),),   
        )
    fig.add_trace(trace2,row=2,col=1)

if df_worker_started.empty == False:
    trace3 = go.Scatter(
        name = 'Worker Started',
        x = df_worker_started['TimeID'], 
        y=df_worker_started['WorkerID'],
        mode = 'markers',
        showlegend=False,
        hovertemplate = 
        '<i>Time: <i> %{x}'+
        '<br><b>Worker:</b>: %{y}<br>',
        marker=dict(
            symbol='circle',
            opacity=1,
            color='lime',
            size=8,
            line=dict(width=1),),
        )
    fig.add_trace(trace3,row=2,col=1)

if df_worker_finished.empty == False:
    trace4 = go.Scatter(
        name = 'Worker Finished',
        x = df_worker_finished['TimeID'], 
        y=df_worker_finished['WorkerID'],
        mode = 'markers',
        showlegend=False,
        hovertemplate = 
        '<i>Time: <i> %{x}'+
        '<br><b>Worker:</b>: %{y}<br>',
        marker=dict(
            symbol='circle',
            opacity=1,
            color='darkorange',
            size=8,
            line=dict(width=1),),
        )
    fig.add_trace(trace4,row=2,col=1)

if df_worker_killed.empty == False:
    trace5 = go.Scatter(
        name = 'Worker killed',
        x = df_worker_killed['TimeID'], 
        y=df_worker_killed['WorkerID'],
        mode = 'markers',
        showlegend=False,
        hovertemplate = 
        '<i>Time: <i> %{x}'+
        '<br><b>Worker:</b>: %{y}<br>',
        marker=dict(
            symbol='circle',
            opacity=1,
            color='orangered',
            size=8,
            line=dict(width=1),),
        )
    fig.add_trace(trace5,row=2,col=1)

fig.update_xaxes(row=2, col=1, nticks = 40, tickangle = 45, tickformat = '%H:%M:%S', title = 'Timestamp', titlefont=dict(size = 20))
fig.update_yaxes(tickmode = 'linear', range = [0.5, max_worker + 0.5], row=2, col=1, title = 'Worker Number', titlefont=dict(size = 20))

fig.update_layout(  template='plotly_dark',
                    autosize=False,
                    width=1000,
                    height=1000,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(
                                orientation="v",
                                y=0.9,
                                x=1),   
                )
# fig2 non paged memory line graph and warning points
fig2 = make_subplots(rows=1, cols=1)

cpus = np.sort(df_non_paged_mem.ComputerID.unique())
cpus = list(map(int, cpus.tolist()))
for cpu in cpus:
    fig2.add_trace(go.Scatter(name = 'Computer-' + str(cpu),
                    x=df_non_paged_mem[df_non_paged_mem['ComputerID'] == cpu]['TimeID'],
                    y = df_non_paged_mem[df_non_paged_mem['ComputerID'] == cpu]['Non_Paged_Mem_MB'],
                    mode='lines',
                    hovertemplate = 
                    '<i>Time: <i> %{x}'+
                    '<br><b>Non Paged Memory [MB]:</b>: %{y}<br>'),
                    row=1,
                    col=1,
                    )

if df_tw_high_non_page.empty == False:
    fig2.add_trace(go.Scatter(
        name = 'High Non Page Memory warnings',
        x = df_tw_high_non_page['TimeID'], 
        y=df_tw_high_non_page['Non_Paged_Mem_MB'],
        mode = 'markers',
        showlegend=False,
        hovertemplate = 
        '<i>Time: <i> %{x}'+
        '<br><b>Non Paged Mem [MB]:</b>: %{y}<br>', 
        marker=dict(
            symbol='x',
            opacity=0.7,
            color='white',
            size=8,
            line=dict(width=1),)
        ),)

fig2.update_xaxes(row=1, col=1, range = [start_time, finish_time], nticks = 40, tickangle = 45, tickformat = '%H:%M:%S', title = 'Timestamp', titlefont=dict(size = 20))
fig2.update_yaxes(row=1, col=1, title = 'Non paged Memory [MB]', titlefont=dict(size = 20), rangemode = 'nonnegative')
fig2.update_layout(template='plotly_dark',)

#fig3 large cluster data transfer warnings

fig3 = make_subplots(rows=1, cols=1)
if df_cluster_warning.empty == False:
    cpus = np.sort(df_cluster_warning.ComputerID.unique())
    cpus = list(map(int, cpus.tolist()))
    for cpu in cpus:
        fig3.add_trace(go.Scatter(name = 'Computer-' + str(cpu),
                        x=df_cluster_warning[df_cluster_warning['ComputerID'] == cpu]['TimeID'],
                        y = df_cluster_warning[df_cluster_warning['ComputerID'] == cpu]['ComputerID'],
                        mode = 'markers',
                        marker=dict(
                            symbol='x',
                            opacity=1,
                            color='darkorange',
                            size=8,),
                        ))
    fig3.update_xaxes(row=1, col=1,  range = [start_time, finish_time], nticks = 40, tickangle = 45, tickformat = '%H:%M:%S', title = 'Timestamp', titlefont=dict(size = 20))
else:
    fig3.update_xaxes(showticklabels=False)

fig3.update_yaxes(tickmode = 'linear', range = [0.5, max_worker + 0.5], row=1, col=1, title = 'Worker Number', titlefont=dict(size = 20))

fig3.update_layout(template='plotly_dark', )

#styles 
colors = {'background': '#111111', 'text': '#7FDBFF'}

text_style = {'textAlign': 'center',
              'color': colors['text']}

button_style = {'textAlign': 'center',
                'color': colors['text'],
                'backgroundColor': '#2F4F4F',
                'border': '2px solid #7FDBFF',
                'padding': '10px 10px',
                'border-radius': '8px',
                'font-size': '14px',
                'margin': '10px 10px',}

tabs_styles = {'width': '300px'}

tab_style = {'textAlign': 'center',
            'color': colors['text'], 
            'backgroundColor': colors['background']}

tab_selected_style = {'color': colors['text'], 'backgroundColor': '#2F4F4F' }

# html layout
app.layout = (html.Div(style={'backgroundColor': colors['background']}, children = [
    dcc.Tabs(
        children = [
        #tab for heatmap
            dcc.Tab(label = 'Panic Heatmap', 
                    children =[
                    html.H1(children='Panic Heatmap',
                            style= text_style
                                ),
                    html.Div(
                        dcc.Graph(id = 'graph', figure=fig)),

                    html.Div([
                        dcc.Markdown("Click on x points in the graph to show details.",
                                    style= text_style),
                        html.Pre(id='click-data'),
                        html.P(
                        id = 'time_worker_details',
                        style= text_style
                            ),

                    html.Div(id = 'table1_div', children =[
                        dcc.Store(id="sub_data"),
                        dcc.Store(id= 'xy_string'),
                        html.Button('Export to PVL', id= 'pvl_button', n_clicks=0,
                                    style= button_style),            
                        html.Table(id = 'composite_table'),
                        dcc.Download(id='df_sub_export'),
                            ])
                            ])
                    ],style = tab_style, selected_style = tab_selected_style
                    ),
            #tab for non paged memory
            dcc.Tab(label= 'Non Paged Memory',
                    children =[ 
                    html.H1(children='Non Paged Memory',
                            style= text_style),
                    html.Div(id="tab_container", 
                        children= [
                        html.Div(id="non_paged_graph_container", 
                        children= [
                        dcc.Graph(id= 'non_paged_graph', figure = fig2),
                        html.P(id = 'time_worker_details2',
                                style= text_style
                            ),        

                        html.Div(id = 'table2_div', children =[
                        dcc.Store(id="sub_data2"),
                        dcc.Store(id= 'xy_string2'),
                        html.Button('Export to PVL', id= 'non_paged_mem_button', n_clicks=0,
                                    style= button_style),       
                        html.Table(id = 'non_paged_composite_table'),
                        dcc.Download(id='df_sub_export2'),
                            ]),
                        ]),

                        html.Div(id = 'cluster_data_graph_container',
                        children = [
                            html.H2(children = 'Cluster Data Warnings',
                                        style= text_style),
                            dcc.Graph(id = 'cluster_data_figure', figure = fig3)
                            ]),
                            html.P(id = 'time_worker_details3',
                                    style= text_style),      

                            html.Div(id = 'table3_div', children = [
                            dcc.Store(id="sub_data3"),
                            dcc.Store(id= 'xy_string3'),
                            # html.Button('Export to PVL', id= 'cluster_data_button', n_clicks=0,
                            #             style= button_style),       
                            html.Table(id = 'cluster_data_table'),
                            dcc.Download(id='df_sub_export3'),
                                ]),
                        ])
                    ], style = tab_style, selected_style = tab_selected_style
                    ),
         ], style = tabs_styles),
    ])
)

#method to replace empty None rules with empty string
def none_to_empty_str(items):
    return {k: v if v is not None else '' for k, v in items}

#callback for xy coordinates, tab 1 graph1
@app.callback(
    Output('time_worker_details', 'children'),
    Output('composite_table', 'children'),
    Output('sub_data', 'data'),
    Output('xy_string', 'data'),
    Input('graph', 'clickData'),
    prevent_initial_call=True
    )

def Update_data_table(clickData):
    if clickData:        
        #get coordinates of click for x,y
        click_details_dict = clickData['points'][0]
        x = str(click_details_dict['x'])
        y = str(click_details_dict['y'])
        xy = x + ', ' + y  
        xy_string = 'Worker' + y + '_' + x[5:]
        #create df of selected xy
        df = df_composites[[xy]]
        df_sub = df_rules['CompositeID'].isin(df[xy])#filter selected compositeId by selected 
        df_sub_table = dash_table.DataTable(df_rules[df_sub].drop_duplicates(subset = ["CompositeID"]).to_dict('records'),#return as a table 
                                            style_cell={'textAlign': 'left'},
                                            style_header={'backgroundColor': 'rgb(30, 30, 30)','color': 'white'},
                                            style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}, 
                                            )
        df_sub = df_rules[df_sub].drop_duplicates(subset = ["CompositeID"]).to_json()
        return xy_string, df_sub_table, df_sub, xy_string
    else:
        raise PreventUpdate
#callback for tab 1, download button 1, table 1
@app.callback(
    Output('df_sub_export', 'data'),
    Output('pvl_button', 'n_clicks'),
    Input('pvl_button', 'n_clicks'),
    Input('sub_data', 'data'),
    Input('xy_string', 'data'),
    prevent_initial_call=True
)
def Export_to_pvl(n_clicks, sub_data, xy_string):
    if n_clicks == 0:
        raise PreventUpdate
    elif sub_data is None and n_clicks > 0:
        n_clicks = 0
        return None, n_clicks
    else: 
        xy_string = str(xy_string)
        dff_dict = json.loads(sub_data, object_pairs_hook=none_to_empty_str)
        string_out = ''
        comp_ids = dff_dict['CompositeID'].values()
        engine_names = dff_dict['EngineName'].values()
        rules = dff_dict['Rules'].values()
        for comp_id, engine_name, rule in zip(comp_ids, engine_names, rules):
            sel =('// ' + str(comp_id) + ' ' + engine_name + '\n' + 'select_check -drc ' + rule + '\n')
            string_out += sel
        n_clicks = 0
        return dict(content= string_out, filename= xy_string + '.pvl'), n_clicks

#callback for xy coordinates, tab 2 graph 1
@app.callback(
     Output('time_worker_details2', 'children'),
     Output('non_paged_composite_table', 'children'),
     Output('sub_data2', 'data'),
     Output('xy_string2', 'data'),
     Input('non_paged_graph', 'clickData')
)
def update_non_paged_table(clickData):
    if clickData:        
        click_details_dict = clickData['points'][0]
        x = str(click_details_dict['x'])
        y = str(click_details_dict['y'])
        xy = x + ', ' + y  
        xy_string = x[5:]
        df = df_high_non_page_composites[[xy]]
        df_sub = df_rules['CompositeID'].isin(df[xy])#filter selected compositeId by selected 
        df_sub_table = dash_table.DataTable(df_rules[df_sub].drop_duplicates(subset = ["CompositeID"]).to_dict('records'),#return as a table 
                                            style_cell={'textAlign': 'left'},
                                            style_header={'backgroundColor': 'rgb(30, 30, 30)','color': 'white'},
                                            style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}, 
                                            )
        
        df_sub = df_rules[df_sub].drop_duplicates(subset = ["CompositeID"]).to_json()
        return xy_string, df_sub_table, df_sub, xy_string
    else:
        raise PreventUpdate
#callback for tab 2, table 1, download button 1
@app.callback(
    Output('df_sub_export2', 'data'),
    Output('non_paged_mem_button', 'n_clicks'),
    Input('non_paged_mem_button', 'n_clicks'),
    Input('sub_data2', 'data'),
    Input('xy_string2', 'data'),
    prevent_initial_call=True
)
def Export_to_pvl(n_clicks, sub_data2, xy_string2):
    if n_clicks == 0:
        raise PreventUpdate
    elif sub_data2 is None and n_clicks > 0:
        n_clicks = 0
        return None, n_clicks
    else:
        xy_string2 = str(xy_string2)
        dff_dict = json.loads(sub_data2, object_pairs_hook=none_to_empty_str)
        string_out = ''
        comp_ids = dff_dict['CompositeID'].values()
        engine_names = dff_dict['EngineName'].values()
        rules = dff_dict['Rules'].values()
        for comp_id, engine_name, rule in zip(comp_ids, engine_names, rules):
            sel =('// ' + str(comp_id) + ' ' + engine_name + '\n' + 'select_check -drc ' + rule + '\n')
            string_out += sel
        n_clicks = 0
        return dict(content= string_out, filename= xy_string2 + '.pvl'), n_clicks

#callback for xy coordinates, tab 2 graph 2
@app.callback(
    Output('time_worker_details3', 'children'),
    Output('cluster_data_table', 'children'),
    Output('sub_data3', 'data'),
    Output('xy_string3', 'data'),
    Input('cluster_data_figure', 'clickData')
)
def update_warning_table(clickData):
    if clickData:
        click_details_dict = clickData['points'][0]
        x = click_details_dict['x']
        y = click_details_dict['y']
        xy_string = str(x) + ', Worker:' + str(y)
        xy_string = xy_string[10:]
        df_sub = pd.DataFrame(df_cluster_warning.loc[df_cluster_warning['TimeID'].eq(x) & df_cluster_warning['ComputerID'].eq(y), 'WarningID'])
        df_sub_table = dash_table.DataTable(df_sub.to_dict('records'),
                                            style_cell={'textAlign': 'left'},
                                            style_header={'backgroundColor': 'rgb(30, 30, 30)','color': 'white'},
                                            style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}, 
                                            )
        df_sub = df_sub.to_json()
        return xy_string, df_sub_table, df_sub, xy_string
    else:
        raise PreventUpdate

p = random.randint(1001, 8999)
app.run_server(port = p, debug=False, use_reloader=False)