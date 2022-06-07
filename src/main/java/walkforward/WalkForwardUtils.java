package walkforward;

import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.CSVWriter;
import com.opencsv.exceptions.CsvException;
import com.opencsv.exceptions.CsvValidationException;
import featureselection.FeatureSelection;
import myexception.NumberOfReleaseOutOfBoundException;
import utils.Dataset;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.core.Instances;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class WalkForwardUtils {
    private WalkForwardUtils() {}

    public static void writeHeader(PrintWriter writer, String datasetName) {
        writer.println("@relation " + datasetName);
        writer.println("@attribute release numeric");
        writer.println("@attribute class string");
        writer.println("@attribute size numeric");
        writer.println("@attribute LOC_touched numeric");
        writer.println("@attribute NR numeric");
        writer.println("@attribute NAuth numeric");
        writer.println("@attribute LOC_added numeric");
        writer.println("@attribute MAX_LOC_added numeric");
        writer.println("@attribute AVG_LOC_added numeric");
        writer.println("@attribute churn numeric");
        writer.println("@attribute MAX_churn numeric");
        writer.println("@attribute buggy {no, yes}");
        writer.println("@data");
    }

    public static int getTrainingSetSize(File dataset, int nRun) throws NumberOfReleaseOutOfBoundException {
        int size = 0;

        try(CSVReader csvReader = new CSVReaderBuilder(new FileReader(dataset)).withSkipLines(1).build()) {
            List<String[]> allRows = csvReader.readAll();
            List<String> releaseScanned = new ArrayList<>();
            for(String[] row : allRows) {
                if(!releaseScanned.contains(row[0])) {
                    releaseScanned.add(row[0]);
                }

                if(releaseScanned.size() > nRun) break;
                size++;

                if(size >= allRows.size()) {
                    throw new NumberOfReleaseOutOfBoundException("Number of the run is greater than the number of the versions. " +
                            "Versions scanned: " + releaseScanned.size() + "nRun: " + nRun);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (CsvException e) {
            throw new RuntimeException(e);
        }

        return size;
    }

    public static void writeRows(PrintWriter writer, List<String[]> allRows) {
        for (String [] row : allRows) {
            int i;
            for(i = 0; i < row.length - 1; i++) {
                if(Objects.equals(row[i], "null")) {
                    writer.print("0,");
                } else {
                    writer.print(row[i] + ",");
                }
            }
            if(Objects.equals(row[i], "null")) {
                writer.println("0");
            } else {
                writer.println(row[i]);
            }
        }
    }

    public static List<String[]> readLines(CSVReader csvReader, int numOfLinesToRead) throws CsvValidationException, IOException {
        List<String[]> firstNLines = new ArrayList<>();
        for(int i = 0; i < numOfLinesToRead; i++) {
            firstNLines.add(csvReader.readNext());
        }
        return firstNLines;
    }

    public static int getNumberOfVersions(File dataset) {
        List<String> releaseScanned = new ArrayList<>();

        try(CSVReader csvReader = new CSVReaderBuilder(new FileReader(dataset)).withSkipLines(1).build()) {
            List<String[]> allRows = csvReader.readAll();
            for(String[] row : allRows) {
                if(!releaseScanned.contains(row[0])) {
                    releaseScanned.add(row[0]);
                }
            }

        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (CsvException e) {
            throw new RuntimeException(e);
        }

        return releaseScanned.size();
    }

    public static int getTestingSetSize(File dataset, int trainingSetSize) {
        List<String> scannedVersion = new ArrayList<>();
        int testingSetSize = 0;

        try(CSVReader csvReader = new CSVReaderBuilder(new FileReader(dataset)).withSkipLines(1).build()) {
            csvReader.skip(trainingSetSize);
            List<String[]> allRows = csvReader.readAll();
            for(String[] row : allRows) {
                if(!scannedVersion.contains(row[0])) {
                    scannedVersion.add(row[0]);
                }
                if(scannedVersion.size() > 1) break;;

                testingSetSize++;
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (CsvException e) {
            throw new RuntimeException(e);
        }

        return testingSetSize;
    }

    public static File removeReleases(String datasetPath, int releaseToRemoveFromTop, int releaseToRemoveFromBottom) {
        File dataset = new File(datasetPath);
        File outputFile = new File(Dataset.CUT_DATASET_CSV);
        int totalReleases = WalkForwardUtils.getNumberOfVersions(dataset);
        List<String> scannedRelease = new ArrayList<>();

        try(CSVReader csvReader = new CSVReaderBuilder(new FileReader(dataset)).build();
        CSVWriter csvWriter = new CSVWriter(new FileWriter(outputFile))) {

            String [] header = csvReader.readNext();

            String[] line;
            /* skipping first rows */
            while(scannedRelease.size() <= releaseToRemoveFromTop) {
                line = csvReader.readNext();
                if(!scannedRelease.contains(line[0])) {
                    scannedRelease.add(line[0]);
                }
            }
            csvWriter.writeNext(header);
            int releaseToWrite = totalReleases - releaseToRemoveFromTop - releaseToRemoveFromBottom;
            scannedRelease.clear();
            while(scannedRelease.size() <= releaseToWrite) {
                line = csvReader.readNext();
                csvWriter.writeNext(line);
                if(!scannedRelease.contains(line[0])) {
                    scannedRelease.add(line[0]);
                }
            }
            csvWriter.flush();

        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (CsvValidationException e) {
            throw new RuntimeException(e);
        }

        return outputFile;
    }

    public static Remove createRemoveFilter(Instances dataset, int[] indices) {
        Remove filter = new Remove();
        filter.setAttributeIndicesArray(indices);
        try {
            filter.setInputFormat(dataset);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return filter;
    }

    public static float computePctDataOnTraining(int trainingSetSize, int totalInstances) {
        return (float) trainingSetSize / totalInstances;
    }

    public static float computePctDefectiveOnDataset(Instances dataset, int datasetSize) {
        int defectiveInstances = dataset.attributeStats(dataset.numAttributes() - 1).nominalCounts[1];
        return (float) defectiveInstances / datasetSize;
    }

    public static void writeCSVHeader(String outputFile) {
        try(CSVWriter csvWriter = new CSVWriter(new FileWriter(outputFile))) {
            String [] header = {"dataset", "#Training Release", "%Training", "%Defective in Training", "%Defective in Testing", "Classifier", "Balancing",
                    "Feature Selection", "Cost Sensitivity", "TP", "FP", "TN", "FN", "precision", "recall", "AUC", "kappa"};
            csvWriter.writeNext(header);
            csvWriter.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void writeResultLine(String outputFile, String dataset, int nTrainingRelease, float pctDataOnTraining, float pctDefectiveInTraining,
                                       float pctDefectiveInTesting, String classifier, String balancing, String featureSelection, String costSensitivity,
                                       Evaluation eval) {

        try(CSVWriter csvWriter = new CSVWriter(new FileWriter(outputFile, true))) {
            double truePositive = eval.numTruePositives(1);
            double falsePositive = eval.numFalsePositives(1);
            double trueNegative = eval.numTrueNegatives(1);
            double falseNegative = eval.numFalseNegatives(1);
            double precision = eval.precision(1);
            double recall = eval.recall(1);
            double auc = eval.areaUnderROC(1);
            double kappa = eval.kappa();
            String[] line = {dataset, String.valueOf(nTrainingRelease), String.valueOf(pctDataOnTraining), String.valueOf(pctDefectiveInTraining),
                    String.valueOf(pctDefectiveInTesting), classifier, balancing, featureSelection, costSensitivity, String.valueOf(truePositive),
                    String.valueOf(falsePositive), String.valueOf(trueNegative), String.valueOf(falseNegative), String.valueOf(precision),
                    String.valueOf(recall), String.valueOf(auc), String.valueOf(kappa)};

            csvWriter.writeNext(line);
            csvWriter.flush();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    public static Evaluation attributeSelection(Instances trainingSet, Instances testingSet, AbstractClassifier classifier) {
        AttributeSelection attributeSelection = FeatureSelection.createBestFirstFilter(trainingSet);
        Evaluation evaluation;
        try {
            Instances filteredTrainingSet = FeatureSelection.createFilteredInstances(trainingSet, attributeSelection);
            Instances filteredTestingSet = FeatureSelection.createFilteredInstances(testingSet, attributeSelection);
            classifier.buildClassifier(filteredTrainingSet);
            evaluation = new Evaluation(filteredTrainingSet);
            evaluation.evaluateModel(classifier, filteredTestingSet);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return evaluation;
    }

    public static Evaluation simpleClassify(Instances trainingSet, Instances testingSet, AbstractClassifier classifier) {
        Evaluation eval;
        try {
            classifier.buildClassifier(trainingSet);
            eval = new Evaluation(trainingSet);
            eval.evaluateModel(classifier, testingSet);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return eval;
    }

    public static int countLinesCSV(File file) {
        List<String[]> allRows;
        try(CSVReader csvReader = new CSVReaderBuilder(new FileReader(file)).withSkipLines(1).build()) {
            allRows = csvReader.readAll();
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (CsvException e) {
            throw new RuntimeException(e);
        }
        return allRows.size();
    }

    public static CostMatrix createCostMatrix(double weightFalsePositive, double weightFalseNegative) {
        CostMatrix costMatrix = new CostMatrix(2);
        costMatrix.setCell(0, 0, 0.0);
        costMatrix.setCell(1, 0, weightFalsePositive);
        costMatrix.setCell(0, 1, weightFalseNegative);
        costMatrix.setCell(1, 1, 0.0);
        return costMatrix;
    }

    public static Evaluation costSensitivityEval(Instances trainingSet, Instances testingSet, AbstractClassifier classifier, double costFalsePositive,
                                                 double costFalseNegative) {
        Evaluation eval;
        try {
            CostSensitiveClassifier costSensitiveClassifier = new CostSensitiveClassifier();
            costSensitiveClassifier.setClassifier(classifier);
            costSensitiveClassifier.setCostMatrix(WalkForwardUtils.createCostMatrix(costFalsePositive, costFalseNegative));
            costSensitiveClassifier.buildClassifier(trainingSet);

            eval = new Evaluation(trainingSet, costSensitiveClassifier.getCostMatrix());
            eval.evaluateModel(costSensitiveClassifier, testingSet);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return eval;
    }

    public static Evaluation featureSelectionThenCostSensitivity(Instances trainingSet, Instances testingSet, AbstractClassifier classifier,
                                                                 double costFalsePositive, double costFalseNegative) {
        /* Feature Selection */
        AttributeSelection attributeFilter = FeatureSelection.createBestFirstFilter(trainingSet);
        Instances filteredTraining = FeatureSelection.createFilteredInstances(trainingSet, attributeFilter);
        Instances filteredTesting = FeatureSelection.createFilteredInstances(testingSet, attributeFilter);
        /* Cost sensitivity */
        return WalkForwardUtils.costSensitivityEval(filteredTraining, filteredTesting, classifier, costFalsePositive, costFalseNegative);

    }

}
