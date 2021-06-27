import pyspark.sql.functions as F

class ChargePointsETLJob:
    input_path = 'data/input/electric-chargepoints-2017.csv'
    output_path = 'data/output/chargepoints-2017-analysis'

    def __init__(self):
        self.spark_session = (SparkSession.builder
                                          .master("local[*]")
                                          .appName("ElectricChargePointsETLJob")
                                          .getOrCreate())

    def extract(self):
        df = self.spark_session.read.option("header", True)\
        .csv('data/input/electric-chargepoints-2017.csv')
        return df

    def transform(self, df):
        df = df.selectExpr('CPID as chargepoint_id', 'PluginDuration as plugin') #이름까지 바꾸려고 selectExpr
        df = df.withColumn("plugin", df["plugin"].cast('float'))
        df = df.groupBy("chargepoint_id")\
        .agg(F.round(F.max("plugin"),2).alias("max_duration")\ 
        , F.round(F.mean("plugin"),2).alias("avg_duration")) #aggregate, alias
        return df

    def load(self, df):
        df = df.write.parquet('data/output/chargepoints-2017-analysis')
        return df

    def run(self):
        self.load(self.transform(self.extract()))