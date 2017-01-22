package org.bigDataCourse.Recommandation;
import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;

public class EvaluateRecommender 
{
    public static void main( String[] args ) throws Exception
    {
    	DataModel model=new FileDataModel(new File(args[0]));
    	RecommenderEvaluator MAEevaluator=new AverageAbsoluteDifferenceRecommenderEvaluator();
    	RecommenderEvaluator RMSevaluator=new RMSRecommenderEvaluator();
    	RecommenderIRStatsEvaluator evaluator = new GenericRecommenderIRStatsEvaluator();
    	RecommenderBuilder builder=new MyRecommenderBuilder();
    	System.out.println("Average Absolute Error "+MAEevaluator.evaluate(builder,null,model, 0.9,1.0));
        System.out.print("Root Mean Square Error "+ RMSevaluator.evaluate(builder, null, model, 0.9, 1.0));
    	IRStatistics stats = evaluator.evaluate(builder,null, model,null, 10,  
    			GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
    	System.out.println("precision: "+stats.getPrecision());
    	System.out.println("Recall: "+stats.getRecall());
    }
}
    class MyRecommenderBuilder implements RecommenderBuilder
    {
		public Recommender buildRecommender(DataModel dataModel) throws TasteException {
			// TODO Auto-generated
			//ItemSimilarity sim1=new EuclideanDistanceSimilarity(dataModel);
			//UserSimilarity sim=new PearsonCorrelationSimilarity(dataModel);
			//UserNeighborhood neighborhood=new ThresholdUserNeighborhood(0.9, sim, dataModel);
			ItemSimilarity similarity=new PearsonCorrelationSimilarity(dataModel);
			return new GenericItemBasedRecommender(dataModel, similarity);
			//return new GenericUserBasedRecommender(dataModel,neighborhood,sim);
		}
    }
