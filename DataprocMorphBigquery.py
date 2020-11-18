from pyspark.sql import SparkSession
from pyspark.sql import types as T, functions as F
from google.cloud import storage
from datetime import datetime as dt
import MeCab
import copy
import jaconv
import pandas as pd


"""
=====================
UDF and utilities
=====================
"""

"""
Tokenizer UDF
"""
class JapaneseTokenizer(object):
    def __init__(self):
        self.mecab = MeCab.Tagger(
            '-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd -u "/usr/lib/x86_64-linux-gnu/mecab/dic/user/neologd_60_noskip.dic"'
        )
        self.mecab.parseToNode('')
 
    def split(self, text):
        node = self.mecab.parseToNode(text)
        words = []
        while node:
            if node.surface:
                words.append(node.surface.decode("UTF-8"))
            node = node.next
        return words

def tokenize(text):
    tokenizer = JapaneseTokenizer()
    return tokenizer.split(text)

def tokenize_and_create_rdd(text):
    return ','.join(tokenize(text.encode("UTF-8")))

tokenize_udf = F.udf(tokenize_and_create_rdd, T.StringType())

"""
str_yyyymmdd_to_date UDF
"""
def str_yyyymmdd_to_date(tdate):
    return '{0}-{1}-{2}'.format(tdate[0:4], tdate[4:6], tdate[6:8])

str_yyyymmdd_to_date_udf = F.udf(str_yyyymmdd_to_date, T.StringType())

"""
get_pre_id_and_post_id UDF
"""
def get_pre_id_and_post_id(idx_id):
    return idx_id.split("_")

get_pre_id_and_post_id_udf = F.udf(get_pre_id_and_post_id, T.ArrayType(T.StringType()))


"""
get_latest_daily_file_name
"""
def get_latest_daily_file_name(bucket_name, file_type, tdate, delimiter=None):
    storage_client = storage.Client()

    prefixCondition = file_type + "/" + file_type + "_" + tdate + "_"
    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefixCondition, delimiter=delimiter
    )

    filename_list = {}
    for blob in blobs:
        file_date_str = blob.name.split('/')[-1].split('_')[-1].split('.')[0]
        tdatetime = dt.strptime(file_date_str, '%Y%m%d%H%M%S')
        filename_list[blob.name] = tdatetime
        
    if any(filename_list):
        latest_file_name = max(filename_list, key=filename_list.get)
        print(latest_file_name)
        return latest_file_name
    else:
        raise Exception("target file '{}' on '{}' is not exist ...".format(file_type, tdate))


"""
=====================
vars
=====================
"""
target_date = "20200929"
target_file_gcs_name = "{target_file_gcs_name}"
target_file_type = "{target_file_type}"

bigquery_dataset = "{bigquery_dataset}"
bigquery_save_table = "{bigquery_save_table}"

try:
    login = pd.read_csv(r'login.txt', header=None)
    user = login[0][0]
    password = login[0][1]
    print('User information is ready!')
except:
    print('Login information is not available!!')

host = '##.##.##.##'
db_name = 'db_name'

cloud_sql_options_base = {
    "url": "jdbc:mysql://{}:5432/{}".format(host, db_name),
    "driver":"com.mysql.jdbc.Driver",
    "user":user,
    "password":password
}


"""
=====================
main
=====================
"""

"""
初期化
"""
spark = SparkSession\
    .builder\
    .master('yarn')\
    .appName('morph-prototype')\
    .getOrCreate()

temp_bucket = "hopstar-dev-dataproc"
spark.conf.set('temporaryGcsBucket', temp_bucket)


"""
Create dataframe from tsv
"""
target_file_name = get_latest_daily_file_name(target_file_gcs_name, target_file_type, target_date)

tsv_gcs_path='gs://{0}/{1}'.format(target_file_gcs_name, target_file_name)
df_tsv = spark.read.csv(tsv_gcs_path, sep=r'\t', header=True)


"""
Create dictionary dataframe from CloudSQL
"""
cloud_sql_options = copy.copy(cloud_sql_options_base)


cloud_sql_options["dbtable"] = "post_keywords"
df_post_keywords_base = spark.read.format("jdbc").options(**cloud_sql_options).load()

"""
Create Keyword
"""

df_post_keywords = df_post_keywords_base.filter(df_post_keywords_base.status == 1).select("post_id", "keyword")

post_dics_dics = {}

post_rows = df_post_keywords.collect()
post_ids = [str(row[0]) for row in post_rows ]
post_ids = list(set(post_ids))

for post_id in post_ids:
    post_dics_dics[post_id] = []
    
for row in post_rows:
    id = str(row[0])
    keyword = str(jaconv.h2z(row[1]).encode("UTF-8").lower())
    post_dics_dics[id].append(keyword)


"""
check_having_the_post_dics_keyword
"""
def check_having_the_post_dics_keyword(post_id, wakati):
    if post_id in post_dics_dics:
        wakati_array = wakati.split(",")
        hit_words = [hit_word for hit_word in wakati_array if jaconv.h2z(hit_word).encode("UTF-8").lower() in post_dics_dics[post_id]]
        if any(hit_words):
            return ",".join(list(set(hit_words)))
        else:
            return None
    else:
        return None

check_having_the_post_dics_keyword_udf = F.udf(check_having_the_post_dics_keyword, T.StringType())




"""
Check
"""
df_check_base = df_tsv.withColumn("wakati", tokenize_udf(F.col("text")))

# df_check_base = df_tsv\
#     .limit(500)\
#     .withColumn("wakati", tokenize_udf(F.col("text")))

df_check = df_check_base\
    .withColumn("ids_array", get_pre_id_and_post_id_udf(F.col("idx_id")))\
    .withColumn("pre_id", F.col("ids_array")[0])\
    .withColumn("post_id", F.col("ids_array")[1])\
    .withColumn("match_post_dics_keywords", check_having_the_post_dics_keyword_udf(F.col("post_id"), F.col("wakati")))\
    .drop("ids_array")\
    .drop("pre_id")\
    .drop("post_id")


"""
Save to BigQuery
"""
df_check_converted = df_check\
    .withColumn("created_at", df_check.created_at.cast(T.TimestampType()))\
    .withColumn("tdate", F.lit(str_yyyymmdd_to_date(target_date)))\
    .withColumn("tdate", F.lit(F.col("tdate").cast("date")))

df_check_converted\
    .write\
    .format('bigquery')\
    .mode('append')\
    .option('table', '{0}.{1}'.format(bigquery_dataset, bigquery_save_table))\
    .option('partitionType', 'DAY')\
    .option('partitionField', 'tdate')\
    .save()
