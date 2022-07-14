#!/grid/common/pkgs/python/v3.9.6/bin/python3.9

import subprocess
subprocess.run('/grid/common/pkgs/python/v3.9.6/bin/pip install dash', shell=True)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash, dash_table, dcc, html
import pathlib
from shlex import join
import sys
import re
from datetime import datetime
from dash.dependencies import Output, Input#
from plotly.callbacks import Points, InputDeviceState
from plotly.subplots import make_subplots
import math

app = dash.Dash()
print('running...')
# 1 read data from pegasus.log, get df composites and df worker started
# 1a: script to extract composite from pegasus.log file, amd return list of composites
#as dataframe and as a string list
def run_composite_extractor(log_file):
    panicking_pattern = re.compile(".*Worker.*is.panicking")
    #running_number_pattern = re.compile("\s*Running:\s\d+")
    pending_pattern = re.compile("\s*Pending:\s\d+")
    rule_pattern = re.compile("\s*\d+:\d+:.*")
    starter_pattern = re.compile(".*Worker\s\d+\sstarted.*")
    flag_panicking = 0
    #time_stamps_list = []
    #worker_stamps_list = []
    time_worker_stamps_list = []
    worker_started_list = []
    composites_list = []
    print('extracting composites...')
    with open(log_file, 'r', encoding= 'latin1') as lines:
        for line in lines:
            if (re.search(panicking_pattern, line) is not None):
                match_time_stamp = re.search(r'.*\s\d+:\d+:\d+\s([0-9]+\.[0-9])+s.*Worker\s(\d+).*', line)
                time_string = str(int(math.ceil(float(match_time_stamp.group(1)))))
                worker_num = match_time_stamp.group(2)
                #worker_stamps_list.append(worker_num)
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
            elif(re.search(starter_pattern, line) is not None):
                started_status = re.search(r'\d+\-\d+\s\d+:\d+:\d+\s([0-9]+\.[0-9])+s.*Worker\s(\d+)\sstarted.*', line)
                time_string = int(math.ceil(float(started_status.group(1))))
                started_worker = started_status.group(2)
                worker_started_list.append((time_string, started_worker))
            else:
                pass     
        #merge the list of time/worker stamps and composites
        merged_lists = list(map(lambda x, y: (x,y), time_worker_stamps_list, composites_list ))
        #create a dict from the tuples - K:V = time/worker: rule
        dict_of_merged = dict(merged_lists)
        #create dataframe with time/worker as column headers
        df_composites = pd.DataFrame({key:pd.Series(value) for key, value in dict_of_merged.items() })
        df_worker_started = pd.DataFrame(worker_started_list, columns =['TimeID', 'WorkerID'])

        #method to get a list of unique composites, avoid repitition
        def flatten_unique_list(extracted_list):
            set_comps =  {i for b in [[i] if not isinstance(i, list) else flatten_unique_list(i) for i in extracted_list] for i in b}
            return list(set_comps)

        extracted_unique_list = flatten_unique_list(composites_list)
        print('composites extracted')
        return df_composites, extracted_unique_list, df_worker_started

#input LOGS and decrypt
input_log_location = str(sys.argv[1])
pegasus_log_file_input = pathlib.Path(input_log_location)

#1b: decrypt logs file
print('decrypting log file...')
output_directory = input_log_location + '.dec/pegasus.log'
subprocess.run('mkdir ' + input_log_location + '.dec', shell=True)
subprocess.run('/vols/ssvrd_t1_007/meg/commander/workspace/hourly_opt/gcc_v9/latest/output/cdn_hier/lnx86_OPT_gcc_v9/tools.lnx86/bin/pegasus -decrypt '+ input_log_location + '/pegasus.log ' + output_directory, shell= True, )
decrypted_string = input_log_location + '.dec/pegasus.log'
decrypted_location = pathlib.Path(decrypted_string)

df_composites, extracted_composites_list, df_worker_started = run_composite_extractor(decrypted_location)

#convert list to string to be passed to script
string_list = str(extracted_composites_list)
string_list = string_list.replace(" ", "").replace("'","").replace("[", "").replace("]", "")
#pass string_list of composites to shell script
#generate stats.rpt.fanout from shell script    #/lan/csv/geomrd1/tools/bin/generate_stats.sh.b71746 --log-dir RUNDIR_22.1/LOGS/ --pegasus /vols/ssvrd_t3nb_001/PVS15.10/geom/lnx86/64/workspace/topo_opt/gcc_v9/hourly_opt.2022-04-26_133516/output/cdn_hier/lnx86_OPT_gcc_v9/tools.lnx86/bin/pegasus --skip-reports --report-fanout-rules
print('Generating Rules Report...')
subprocess.run('/lan/csv/geomrd1/tools/bin/generate_stats.sh.b71746 --log-dir ' + input_log_location + '/ --pegasus /vols/ssvrd_t1_007/meg/commander/workspace/hourly_opt/gcc_v9/latest/output/cdn_hier/lnx86_OPT_gcc_v9/tools.lnx86/bin/pegasus --skip-reports --report-fanout-rules ' + string_list, shell = True)

print('Rules Report generated')
#1c get stats.rpt.fanout file
rules_file_input = pathlib.Path('reports/stats.rpt.fanout')

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
        #rules_list is all the rules in one file, DRc_list is a nested list of rules with composite name
        return df
    except FileNotFoundError:
        print('File not accessible')

print('Extracting rules from rules.rpt.fanout')
df_rules = get_rules(rules_file_input)
print('Rules extracted')
#2a read data from usr.log, get df usr log panic, finished time
#method for getting the times when panics overlap
def run_usr_log_extractor(usr_log):
    status_pattern = re.compile("^INFO:.*Status:.*Elapsed:.")
    heading_pattern = re.compile("\s*Worker:\s*Active\s*Threads")
    panic_pattern = re.compile("\s*\d+:\s*Panic\s.*")
    data_pattern = re.compile("\s*\d+:\s+\d+")
    flag_status = 0
    flag_worker = 0
    list_worker_time = []
    with open(usr_log, 'r') as lines:
        for line in lines:
            if re.search(status_pattern, line) is not None:
                flag_status = 1
                match_status = re.search(r'[\d:]+\s*Elapsed:\s+(\d+:\d+:\d+)+', line)
                time_string = match_status.group(1)
                finished_string = match_status.group(1)
            elif (re.search(heading_pattern, line) is not None) and (flag_status == 1):
                flag_status = 1
                flag_worker = 1
            elif (re.search(data_pattern, line) is not None) and (flag_status == 1) and (flag_worker == 1):
                pass
            elif (re.search(panic_pattern, line) is not None) and (flag_status == 1) and (flag_worker == 1):
                #search for data string pattern, and put digits into groups - each (...) is a grou
                match_worker_details = re.search(r'\s*(\d+):\s*Panic.', line)
                worker_id = int(match_worker_details.group(1))
                pt = datetime.strptime(time_string,'%H:%M:%S')
                total_seconds = pt.second + pt.minute*60 + pt.hour*3600
                dict1 = {"TimeID": total_seconds, "WorkerID": worker_id}
                list_worker_time.append(dict1)
            else:
                flag_status = 0
                flag_worker = 0
        pt = datetime.strptime(finished_string,'%H:%M:%S')
        finish_time = pt.second + pt.minute*60 + pt.hour*3600
        
        df_usr = pd.DataFrame(list_worker_time)
        return df_usr, finish_time

print('Extracting from usr.log')
df_usr, finish_time = run_usr_log_extractor('reports/usr.log')

print('usr.log done')

#3 merge pegasusdf and usr_df
def get_xy_for_composites(df_composites):
    col_list = df_composites.columns.values.tolist()
    #convert list of strings to list of tuples
    col_list = [tuple(map(str, sub.split(', '))) for sub in col_list]
    df_xy_comp = pd.DataFrame(col_list, columns = ['TimeID', 'WorkerID'])
    df_xy_comp['TimeID'] = df_xy_comp['TimeID'].astype(int)
    df_xy_comp['WorkerID'] = df_xy_comp['WorkerID'].astype(int)#convert worker obj to int
    return df_xy_comp


df_tw_pegasus = get_xy_for_composites(df_composites)
# print('started',type(df_worker_started['TimeID'][0]), df_worker_started['TimeID'][0])
# print('df_usr time',type(df_usr['TimeID'][0]), df_usr['TimeID'][0])
# print('df_tw time',type(df_tw_pegasus['TimeID'][0]), df_tw_pegasus['TimeID'][0])
# print('df_usr worker',type(df_usr['WorkerID'][0]), df_usr['WorkerID'][0])
# print('df_tw worker',type(df_tw_pegasus['WorkerID'][0]), df_tw_pegasus['WorkerID'][0])

df_outer_merge = pd.merge(df_tw_pegasus, df_usr, how='outer', on=['TimeID', 'WorkerID']).sort_values('TimeID')
start_time = df_worker_started['TimeID'][0]

#all the extracted rules   
print('tracing graph')
# 5 trace graphs
trace1 = go.Histogram2d(
    x=df_outer_merge['TimeID'], 
    y=df_outer_merge['WorkerID'],
    colorscale='oryel',
    xbins= dict(start = start_time, end = finish_time, size = 60),#size = 60000
    #ybins = dict(start = 0, end = worker_max + 1, size = 1),
    zauto=True,
    hovertemplate = 
    '<i>Minute: <i> %{x}'+
    '<br><b>Worker:</b>: %{y}<br>', 
    )
#create second trace, giving exact x,y of panic
trace2 = go.Scatter(
    x = df_tw_pegasus['TimeID'], 
    y=df_tw_pegasus['WorkerID'],
    mode = 'markers',
    # xhoverformat= '%X',
    hovertemplate = 
    '<i>Time: <i> %{x}'+
    '<br><b>Worker:</b>: %{y}<br>', 
    marker=dict(
        symbol='x',
        opacity=0.7,
        color='white',
        size=8,
        line=dict(width=1),
    )  
    )
trace3 = go.Scatter(
    x = df_worker_started['TimeID'], 
    y=df_worker_started['WorkerID'],
    mode = 'markers',
    # xhoverformat= '%X',
    hovertemplate = 
    '<i>Time: <i> %{x}'+
    '<br><b>Worker:</b>: %{y}<br>', 
    marker=dict(
        symbol='circle',
        opacity=0.7,
        color='blue',
        size=8,
        line=dict(width=1),
    )
    )

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(trace1)
fig.add_trace(trace2,secondary_y=True)
fig.add_trace(trace3)

fig.update_layout(
    xaxis = dict(
    tickmode = 'auto', # can also be array or linear
    range = [start_time, finish_time],
    showspikes = True
        ),
    yaxis = dict(
        title_text = 'Worker Number',
        tickmode = 'linear',
        tick0 = 0.0,
        dtick = 1.0,  
        ),
    hovermode='x unified'#clear hover line on x axis
    )

app.layout = html.Div([
        html.Div(
    dcc.Graph(id = 'graph', figure=fig)),
            html.Div([
            dcc.Markdown("Click on x points in the graph to show details."),
            html.Pre(id='click-data'),
        ]),
    html.Div(
        id = 'time_worker_details',
            ),
    html.Div(
        id = 'composite_table',
            )
    ])

@app.callback(
    Output('time_worker_details', 'children'),
    Output('composite_table', 'children'),
    Input('graph', 'clickData'))

def Update_data_table(clickData):
    if clickData:        
        #get coordinates of click for x,y
        click_details_dict = clickData['points'][0]
        x = str(click_details_dict['x'])
        y = str(click_details_dict['y'])
        xy = x + ', ' + y  
        #create df of selected xy
        df = df_composites[[xy]]
        df_sub = df_rules['CompositeID'].isin(df[xy])#filter selected compositeId by selected 
        df_sub_table = dash_table.DataTable(df_rules[df_sub].to_dict('records'),#return as a table 
        style_cell={'textAlign': 'left'},
        export_format='csv',
        export_headers='display')
        return xy, df_sub_table

#print('Dash is running on http://127.0.0.1:8050/')  
app.run_server(debug=True, use_reloader=False)
# if __name__ == '__main__':
#     app.run_server(debug = True)

