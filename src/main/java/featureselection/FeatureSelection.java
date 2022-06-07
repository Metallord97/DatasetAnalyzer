package featureselection;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.Logger;

public class FeatureSelection {
    private static final Logger LOGGER = LogManager.getLogManager().getLogger(FeatureSelection.class.getName());
    private FeatureSelection(){}

    public static AttributeSelection createBestFirstFilter(Instances trainingSet) {
        AttributeSelection filter = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search = new BestFirst();
        filter.setEvaluator(eval);
        filter.setSearch(search);
        try {
            filter.setInputFormat(trainingSet);
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Exception caught", e);
        }

        return filter;
    }

    public static Instances createFilteredInstances(Instances instances, AttributeSelection filter) {
        Instances filteredInstances = null;
        try {
            filteredInstances = Filter.useFilter(instances, filter);
            int numAttr = filteredInstances.numAttributes();
            filteredInstances.setClassIndex(numAttr - 1);
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Exception caught", e);
        }
        return filteredInstances;
    }


}
