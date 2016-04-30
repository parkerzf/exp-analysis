import psycopg2
import getpass

import pandas as pd
import numpy as np

import plotly as py
from plotly import tools
from plotly.graph_objs import Bar, Layout


#############################################################
#################### user configurations ####################
#############################################################
host = 'r5-lx-l2.zai'
database = "booking"
username = 'feng.zhao' # getpass.getuser() # login username
pwd = getpass.getpass()

# table to be analysed
table       = 'overall'
class_col   = 'hotel_cluster'
# column with set(train/test) info
set_col     = 'set'
# column with booking time (used in time analysis)
time_col    = 'date_time'
# ignore this columns
ignore_cols = 'id, user_location_city, user_id, psp_reference, first_payment_psp_reference, creation_date, is_booking'

folder      = 'plots'
#############################################################
#################### user configurations ####################
#############################################################



#############################################################
######## start connection to db and get metadata ############
#############################################################

try:
   conn = psycopg2.connect("host='%s' dbname='%s' user='%s' password='%s'" % (host, database, username, pwd))
   cur = conn.cursor()
except Exception, e:
   print "Unable to connect to the database: ", str(e)
   exit(-1)

# transform ignore_cols to be used in SQL queries
_ignore_cols = ignore_cols.split(',') + [class_col, time_col, set_col] 
_ignore_cols = "','".join(filter(lambda x: x != "", map(str.strip, _ignore_cols)))

# get categorical columns
cur.execute('''SELECT column_name 
               FROM information_schema.columns 
               WHERE table_schema = 'public' 
                 AND table_name = '%s' 
                 AND data_type in ('character varying','character') 
                 AND column_name not in ('%s')''' % (table, _ignore_cols))

cate_columns = [x[0] for x in cur.fetchall()]

# get numeric columns
cur.execute('''SELECT column_name 
               FROM information_schema.columns 
               WHERE table_schema = 'public' 
                 AND table_name = '%s' 
                 AND data_type in ('integer','smallint','bigint','double precision')
                 AND column_name not in ('%s')''' % (table, _ignore_cols))

nume_columns = [x[0] for x in cur.fetchall()] 

# get datetime columns
cur.execute('''SELECT column_name 
               FROM information_schema.columns 
               WHERE table_schema = 'public' 
                 AND table_name = '%s' 
                 AND data_type in ('date','time','time with time zone','timestamp','timestamp with time zone')
                 AND column_name not in ('%s')''' % (table, _ignore_cols))

time_columns = [x[0] for x in cur.fetchall()] 


def get_info_columns():
   q = ', '.join(['min({0}), max({0}), avg({0})'.format(col) for col in nume_columns])
   cur.execute('SELECT {0} FROM public.{1} WHERE {2} is not null AND is_booking = 1'.format(q, table, set_col))
   r = [x for x in cur.fetchall()[0]] # convert a tuple with multiple entries in an array
   
   d = {} # dict indexed by column's name with the info needed
   # put the stats about numeric columns into dict
   for c, i in zip(nume_columns, range(len(nume_columns))):
      d.update({c: {'min': r[i*3], 'max': r[i*3+1], 'avg': r[i*3+2]}})

   cur.execute('SELECT {0}, count(*), sum({1}) FROM public.{2} WHERE {0} is not null AND is_booking = 1 GROUP BY 1'.format(set_col, class_col, table))
   # the fillna() is used because the test_set have NA label
   df = pd.DataFrame(cur.fetchall()).fillna(0)

   # put another stats into dict
   overall = {'total': df.ix[:,1].sum()}
   for i in range(df.shape[0]):
      overall.update({df.ix[i,0]: df.ix[i,1]})
   
   d.update({'overall': overall})
   return d

info_columns = get_info_columns()

###############################################################
######## end connection to the db and get metadata ############
###############################################################

def do_categ_query(column, top = 20):
   # get the top values with more instances
   cur.execute('''SELECT {0}, count(*) as total_trx 
                  FROM public.{1}
                  WHERE {3} is not null 
                  AND is_booking = 1
                  GROUP BY 1
                  ORDER BY 2 desc 
                  LIMIT {2}'''.format(column, table, top, set_col))

   df1 = pd.DataFrame(cur.fetchall())
   df1.columns = [desc[0] for desc in cur.description]
   
   # remove empty or null values
   #values = "','".join(df1.ix[pd.notnull(df1.ix[:,0]) & (str.strip(df1.ix[:,0]) != ""), 0].values)
   values = "','".join(df1.ix[pd.notnull(df1.ix[:,0]), 0].values)

   # for the top values, breakdown per set_col
   cur.execute('''SELECT {1}, {0}, count(*) as total_trx
                  FROM public.{2}
                  WHERE {0} in ('{4}')
                    AND {1} is not null
                    AND is_booking = 1
                  GROUP BY 1, 2 
                  ORDER BY 1, 2'''.format(column, set_col, table, top, values))

   df2 = pd.DataFrame(cur.fetchall())
   df2.columns = [desc[0] for desc in cur.description]
   # put in columns the different values of set_col
   df2 = df2.pivot(column, set_col, 'total_trx').fillna(0).reset_index()

   return (df1, df2)

def draw_categ(df1, df2, path, column):
   color = ['rgb(31,119,180)', 'rgb(255,127,14)', 'rgb(44,160,44)'] # blue, orange, green
   marker = {}
   for i, c in zip(range(len(df2.columns[1:])), df2.columns[1:]):
      marker.update({c: color[i]})

   ####
   data = []
   url1 = '%s/%s_top.html' % (path, column)
   data.append(Bar(x    = range(df1.shape[0]),
              y    = df1.ix[:,1],
              text = map(lambda x: '{:6.2f}%'.format((100.0 * x) / info_columns['overall']['total']), df1.ix[:,1].values),
              name=''
          ))

   py.offline.plot({
   "data": data,
   "layout": Layout(
      title="Bookings\' distributions",
      yaxis=dict(
         title='Bookings',
      ),
      xaxis=dict(
         title='',
         ticktext = map(lambda x: '%s' % x , df1.ix[:,0]),
         tickvals = range(df1.shape[0])
      )
   )
   }, auto_open=False, filename=url1)
   ####

   ####
   data = []
   # after the second column are the domain values of the breakdown column
   for v in df2.columns[1:]:
     data.append(Bar(x      = range(df2[v].shape[0]), 
                     y      = np.round((100.0 * df2[v].values) / info_columns['overall'][v], 2), 
                     name   = v,
                     marker = dict(color = marker[v]), 
                     text   = map(lambda x: '%s' % int(x), df2[v].values)))
   url2 = '%s/%s_top_break_total.html' % (path, column)
   py.offline.plot({
   "data": data,
   "layout": Layout(
      title="Bookings\'' distribution per set",
      yaxis=dict(
         title='Bookings',
         ticksuffix='%'
      ),
      xaxis=dict(
         title='',
         ticktext = map(lambda x: '%s' % x , df2.ix[:,0]),
         tickvals = range(df2.shape[0])
      ),
      barmode='group'
   )
   }, auto_open=False, filename=url2)
   ####

   return (url1, url2)


def map_column(col):
   print col
   df1, df2 = do_categ_query(col)
   urls = draw_categ(df1, df2, folder, col)
   return (col, urls)


urls = [map_column(c) for c in cate_columns] #map(map_column, cate_columns)



###############################################################
########      generate the final report.html       ############
###############################################################


html_string = '''
<html>
    <head>
       <!-- <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">-->
        <style>body{ margin:0 100; background:whitesmoke; }</style>
    </head>
    <body>
       <h1> Top values </h1>
'''

for col, url in urls:
   html_string += '''
                  <h2>%s</h2>
                  <iframe align="left"  width="50%%" height="550" frameborder="0" seamless="seamless" scrolling="no" src="%s"></iframe>
                  <iframe align="right" width="50%%" height="550" frameborder="0" seamless="seamless" scrolling="no" src="%s"></iframe>
                  ''' % (col, url[0], url[1])

html_string += '''
    </body>
</html>
'''

f = open('report.html','w')
f.write(html_string)
f.close()

conn.close()
