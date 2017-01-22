import org.apache.spark.SparkConf;
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.recommendation._;
object Driver{
  def main(args:Array[String]):Unit =
  {
     if (args.length < 2){
        System.err.println("Usage: q1 <input file>  <output file>")
        System.exit(1);    
      }
   
   val rank=10;val lambda=0.01;val numIterations=20;val k=20;val alpha=0.01;
   var conf=new SparkConf().setAppName("Recommendation").setMaster("yarn");
   conf.set("spark.sql.warehouse.dir","hdfs://hadoopmaster-ravi-pujitha:9000/user/hive/warehouse")
   val sc=new SparkContext(conf);
   val sqlContext= new org.apache.spark.sql.SQLContext(sc)
   import sqlContext.implicits._
   var user_list=sc.textFile(args(0)).map(x=>(x.split("\\s+")(0),x.split("\\s+")(1),x.split("\\s+")(2))).filter(x=>x._1.nonEmpty && x._2.nonEmpty && x._3.nonEmpty).toDF()
   var artistAlias=sc.textFile(args(1)).map(x=>(x.split("\\s+")(0),x.split("\\s+")(1))).filter(x=>x._1.nonEmpty && x._2.nonEmpty).toDF()
   user_list.createOrReplaceTempView("users");
   artistAlias.createOrReplaceTempView("bArtists");
   var finalResult=sqlContext.sql("select distinct t._1 as uId ,t._2 as aId,t._3 as playCount,m._2 as goodId from users t left outer join bArtists m on t._2=m._2").map(x=>{
      var uId:String=x.getString(0)
      var gId:String=x.getString(1)
      var pCount:String=x.getString(2)
      if(x.getString(3)!=null)
      {
      gId=x.getString(3)
      }
      org.apache.spark.mllib.recommendation.Rating(uId.toInt,gId.toInt,pCount.toInt)
      }).rdd
       val percentileAvg=ArrayBuffer[Double]()
      val Array(training,testing)=finalResult.randomSplit(Array(0.8,0.2)) 
      val model= ALS.trainImplicit(training,rank,numIterations,lambda,alpha)
      val actual= testing.map({case org.apache.spark.mllib.recommendation.Rating(a,b,c)=>(a,(b,c))}).groupBy(_._1).mapValues(_.toList).collect({case(a,b)=>b}).flatMap(element => element.map({case(a)=>(a._1,a._2._1,a._2._2)})).sortBy(r=>(r._1,r._2))
      val userproducts=testing.map{case Rating(user,product,rate)=>(user,product)}
      val predictions=model.predict(userproducts).map{case Rating(user,product,rate)=>(user,product,rate)}
      var predict_temp=predictions.groupBy(_._1).mapValues(_.toList).map({case(a,b)=>(b,b.size)}).map({case(a,b)=>(a.map({case(a)=>(b,a._1,a._2,a._3)}))}).map(x=>x.sortBy(_._4).reverse.zipWithIndex.map(element=>(element._1._1,element._1._2,element._1._3,element._1._4,element._2+1)))
      var predictedDF=predict_temp.flatMap({case(z)=>(z.map({z=>(z._2,z._3,z._5.toDouble/z._1)}))}).toDF("userId","productId","count")
      var actualDF=actual.toDF("userId","productId","count")
      actualDF.createOrReplaceTempView("actualDF") 
      predictedDF.createOrReplaceTempView("predictedDF")
      var rank1=sqlContext.sql("select actualDF.userId, actualDF.productId, actualDF.count, predicted.rank from actualDF INNER JOIN predictedDF where actualDF.userId=predicted.userId AND actualDF.productId=predicted.productId")
      val rank2=rank1.rdd.map({case(a)=>(a(0).toString.toInt,a(1).toString.toInt,a(2).toString.toDouble,a(3).toString.toDouble)}).map({case(user,product,count,rank)=>(user,count,count.toDouble*rank)})
      val totalCount=rank2.map({case (user,count,percentileNumerator)=>(user,count)}).reduceByKey(_+_)
      val totalPercentaileRank=rank2.map({case (user,count,percentileNumerator)=>(user,percentileNumerator)}).reduceByKey(_+_)
      var totalCountDF=totalCount.toDF("user","count")
      totalCountDF.createOrReplaceTempView("totalCountDF")
      var totalPercentaileDF=totalPercentaileRank.toDF("user","sum")
      totalPercentaileDF.createOrReplaceTempView("totalPercentaileDF")
      val temp_Result= sqlContext.sql("select count(totalCountDF.user), sum(totalPercentaileDF.sum/totalCountDF.count) as sum from totalCountDF JOIN totalPercentaileDF where totalCountDF.user=totalPercentaileDF.user")
      val Result=temp_Result.rdd.map({case (a)=>(a(0).toString.toInt,a(1).toString.toDouble)}).map({case(a,b)=>(b/a)})
      Result.saveAsTextFile(args(2))
      sc.stop();
  }
}