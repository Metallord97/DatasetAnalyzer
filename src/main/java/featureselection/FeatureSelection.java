package featureselection;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class FeatureSelection {
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
            throw new RuntimeException(e);
        }

        return filter;
    }

    public static Instances createFilteredInstances(Instances instances, AttributeSelection filter) {
        Instances filteredInstances;
        try {
            filteredInstances = Filter.useFilter(instances, filter);
            int numAttr = filteredInstances.numAttributes();
            filteredInstances.setClassIndex(numAttr - 1);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return filteredInstances;
    }


}
