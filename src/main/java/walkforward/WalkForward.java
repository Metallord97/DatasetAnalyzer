package walkforward;

import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.exceptions.CsvException;
import myexception.SetTypeException;
import utils.BookkeeperData;
import utils.StringConstant;
import utils.ZookeeperData;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.*;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.Logger;

public class WalkForward {
    private static final Logger LOGGER = LogManager.getLogManager().getLogger(WalkForward.class.getName());

    private WalkForward() {}

    public static InputStream getSet(File dataset, SetType setType, int trainingSetSize, int testingSetSize) throws SetTypeException {
        InputStream inputStream = null;

        try(CSVReader csvReader = new CSVReaderBuilder(new FileReader(dataset)).withSkipLines(1).build();
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            PrintWriter writer = new PrintWriter(new BufferedOutputStream(byteArrayOutputStream))) {
            /* header */
            WalkForwardUtils.writeHeader(writer, dataset.getName());

            switch (setType) {
                case TRAINING:
                    List<String[]> firstRows = WalkForwardUtils.readLines(csvReader, trainingSetSize);
                    WalkForwardUtils.writeRows(writer, firstRows);
                    break;

                case TESTING:
                    csvReader.skip(trainingSetSize);
                    List<String[]> lastRows = WalkForwardUtils.readLines(csvReader, testingSetSize);
                    WalkForwardUtils.writeRows(writer, lastRows);
                    break;

                default:
                    throw new SetTypeException("SetType must be TRAINING or TESTING");
            }

            writer.flush();
            inputStream = new ByteArrayInputStream(byteArrayOutputStream.toByteArray());

        } catch (FileNotFoundException e) {
            LOGGER.log(Level.SEVERE, "File not found", e);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "I/O exception occurred", e);
        } catch (CsvException e) {
            LOGGER.log(Level.SEVERE, "CsvException occurred", e);
        }
        return inputStream;
    }

    public static void walkForward(String datasetPath, int releaseToRemoveFromTop, int releaseToRemoveFromBottom, String outputFile, String datasetName) throws Exception {
        File datasetFile = WalkForwardUtils.removeReleases(datasetPath, releaseToRemoveFromTop, releaseToRemoveFromBottom);
        int nTrainingRelease = WalkForwardUtils.getNumberOfVersions(datasetFile);
        WalkForwardUtils.writeCSVHeader(outputFile);

        for(int i = 1; i < nTrainingRelease - 1; i++) {
            int trainingSetSize = WalkForwardUtils.getTrainingSetSize(datasetFile, i);
            int testingSetSize = WalkForwardUtils.getTestingSetSize(datasetFile, trainingSetSize);
            DataSource sourceTraining = new DataSource(WalkForward.getSet(datasetFile, SetType.TRAINING, trainingSetSize, testingSetSize));
            DataSource sourceTesting = new DataSource(WalkForward.getSet(datasetFile, SetType.TESTING, trainingSetSize, testingSetSize));
            Instances training = sourceTraining.getDataSet();
            Instances testing = sourceTesting.getDataSet();

            Remove removeFilter = WalkForwardUtils.createRemoveFilter(training, new int[] {0, 1});
            Instances newTraining = Filter.useFilter(training, removeFilter);
            Instances newTesting = Filter.useFilter(testing, removeFilter);

            int numAttr = newTraining.numAttributes();
            newTraining.setClassIndex(numAttr - 1);
            newTesting.setClassIndex(numAttr - 1);

            Evaluation randomForestEvaluation;
            Evaluation naiveBayesEvaluation;
            Evaluation iBkEvaluation;
            int totalInstances = WalkForwardUtils.countLinesCSV(datasetFile);
            float pctDataOnTraining = WalkForwardUtils.computePctDataOnTraining(trainingSetSize, totalInstances);
            float pctDefectiveInTraining = WalkForwardUtils.computePctDefectiveOnDataset(newTraining, trainingSetSize);
            float pctDefectiveInTesting = WalkForwardUtils.computePctDefectiveOnDataset(newTesting, testingSetSize);

            /* NO feature selection */
            randomForestEvaluation = WalkForwardUtils.simpleClassify(newTraining, newTesting, new RandomForest());
            naiveBayesEvaluation = WalkForwardUtils.simpleClassify(newTraining, newTesting, new NaiveBayes());
            iBkEvaluation = WalkForwardUtils.simpleClassify(newTraining, newTesting, new IBk());

            WalkForwardUtils.writeResultLine(outputFile, datasetName, i, pctDataOnTraining, pctDefectiveInTraining,
                    pctDefectiveInTesting, StringConstant.RANDOM_FOREST, StringConstant.NO, StringConstant.NO, StringConstant.NO,
                    randomForestEvaluation);
            WalkForwardUtils.writeResultLine(outputFile, datasetName, i, pctDataOnTraining, pctDefectiveInTraining,
                    pctDefectiveInTesting, StringConstant.NAIVE_BAYES, StringConstant.NO, StringConstant.NO, StringConstant.NO,
                    naiveBayesEvaluation);
            WalkForwardUtils.writeResultLine(outputFile, datasetName, i, pctDataOnTraining, pctDefectiveInTraining,
                    pctDefectiveInTesting, StringConstant.IBK, StringConstant.NO, StringConstant.NO, StringConstant.NO,
                    iBkEvaluation);


            /* Feature Selection */
            randomForestEvaluation = WalkForwardUtils.attributeSelection(newTraining, newTesting, new RandomForest());
            naiveBayesEvaluation = WalkForwardUtils.attributeSelection(newTraining, newTesting, new NaiveBayes());
            iBkEvaluation = WalkForwardUtils.attributeSelection(newTraining, newTesting, new IBk());

            WalkForwardUtils.writeResultLine(outputFile, datasetName, i, pctDataOnTraining, pctDefectiveInTraining,
                    pctDefectiveInTesting, StringConstant.RANDOM_FOREST, StringConstant.NO, StringConstant.YES, StringConstant.NO,
                    randomForestEvaluation);
            WalkForwardUtils.writeResultLine(outputFile, datasetName, i, pctDataOnTraining, pctDefectiveInTraining,
                    pctDefectiveInTesting, StringConstant.NAIVE_BAYES, StringConstant.NO, StringConstant.YES, StringConstant.NO,
                    naiveBayesEvaluation);
            WalkForwardUtils.writeResultLine(outputFile, datasetName, i, pctDataOnTraining, pctDefectiveInTraining,
                    pctDefectiveInTesting, StringConstant.IBK, StringConstant.NO, StringConstant.YES, StringConstant.NO,
                    iBkEvaluation);

            /* Cost Sensitivity */
            randomForestEvaluation = WalkForwardUtils.costSensitivityEval(newTraining, newTesting, new RandomForest(), 1, 10);
            naiveBayesEvaluation = WalkForwardUtils.costSensitivityEval(newTraining, newTesting, new NaiveBayes(), 1, 10);
            iBkEvaluation = WalkForwardUtils.costSensitivityEval(newTraining, newTesting, new IBk(), 1, 10);

            WalkForwardUtils.writeResultLine(outputFile, datasetName, i, pctDataOnTraining, pctDefectiveInTraining,
                    pctDefectiveInTesting, StringConstant.RANDOM_FOREST, StringConstant.NO, StringConstant.NO, StringConstant.YES,
                    randomForestEvaluation);
            WalkForwardUtils.writeResultLine(outputFile, datasetName, i, pctDataOnTraining, pctDefectiveInTraining,
                    pctDefectiveInTesting, StringConstant.NAIVE_BAYES, StringConstant.NO, StringConstant.NO, StringConstant.YES,
                    naiveBayesEvaluation);
            WalkForwardUtils.writeResultLine(outputFile, datasetName, i, pctDataOnTraining, pctDefectiveInTraining,
                    pctDefectiveInTesting, StringConstant.IBK, StringConstant.NO, StringConstant.NO, StringConstant.YES,
                    iBkEvaluation);

            /* Feature Selection & Cost sensitivity */
            randomForestEvaluation = WalkForwardUtils.featureSelectionThenCostSensitivity(newTraining, newTesting, new RandomForest(), 1, 10);
            naiveBayesEvaluation = WalkForwardUtils.featureSelectionThenCostSensitivity(newTraining, newTesting, new NaiveBayes(), 1, 10);
            iBkEvaluation = WalkForwardUtils.featureSelectionThenCostSensitivity(newTraining, newTesting, new IBk(), 1, 10);

            WalkForwardUtils.writeResultLine(outputFile, datasetName, i, pctDataOnTraining, pctDefectiveInTraining,
                    pctDefectiveInTesting, StringConstant.RANDOM_FOREST, StringConstant.NO, StringConstant.YES, StringConstant.YES,
                    randomForestEvaluation);
            WalkForwardUtils.writeResultLine(outputFile, datasetName, i, pctDataOnTraining, pctDefectiveInTraining,
                    pctDefectiveInTesting, StringConstant.NAIVE_BAYES, StringConstant.NO, StringConstant.YES, StringConstant.YES,
                    naiveBayesEvaluation);
            WalkForwardUtils.writeResultLine(outputFile, datasetName, i, pctDataOnTraining, pctDefectiveInTraining,
                    pctDefectiveInTesting, StringConstant.IBK, StringConstant.NO, StringConstant.YES, StringConstant.YES,
                    iBkEvaluation);

        }

    }

    public static void main(String [] args) throws Exception {
        WalkForward.walkForward(ZookeeperData.DATASET, ZookeeperData.RELEASE_TO_REMOVE_FROM_TOP, ZookeeperData.RELEASE_TO_REMOVE_FROM_BOTTOM,
                ZookeeperData.RESULT, ZookeeperData.DATASET_NAME);
        WalkForward.walkForward(BookkeeperData.DATASET, BookkeeperData.RELEASE_TO_REMOVE_FROM_TOP, BookkeeperData.RELEASE_TO_REMOVE_FROM_BOTTOM,
                BookkeeperData.RESULT, BookkeeperData.DATASET_NAME);
    }

}
