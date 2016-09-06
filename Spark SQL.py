
 
irisRDD = SpContext.textFile("iris.csv")
#Remove the header line
irisData = irisRDD.filter(lambda x: "Sepal" not in x)
irisData.count()

#Split the columns
cols = irisData.map(lambda l : l.split(","))
#Make row objects
from pyspark.sql import Row
irisMap = cols.map( lambda p: Row ( SepalLengh = p[0], \
                                   SepalWidth = p[1], \
                                   PetalLength = p[2], \
                                   PetalWidth = p[3], \
                                   Species = p[4] ))
irisMap.collect()
#Create a data frame from the Row objects
irisDF = SpSession.createDataFrame(irisMap)
irisDF.select("*").show()

"""
-----------------------------------------------------------------------------
In the irisDF, filter for rows whose PetalWidth is greater than 0.4
and count them.
Hint: Check for Spark documentation on how to count rows : 
https://spark.apache.org/docs/latest/api/python/pyspark.sql.html
-----------------------------------------------------------------------------
"""
irisDF.filter( irisDF["PetalWidth"] > 0.4).count()    
    
"""
-----------------------------------------------------------------------------
#Spark SQL Temp Tables
***********************

 Register a temp table called "iris" using irisDF. Then find average
Petal Width by Species using that table.
-----------------------------------------------------------------------------
"""
irisDF.registerTempTable("iris")
sqlContext.sql("select Species,avg(PetalWidth) from iris group by Species")\
.show()

irisDF.createOrReplaceTempView("iris")
SpSession.sql("select Species,avg(PetalWidth) from iris group by Species")\
.show()

"""
-----------------------------------------------------------------------------
Hope you had some good practice !! Recommend trying out your own use cases
-----------------------------------------------------------------------------
"""
    



