package utils;

public class CSVRow {
    private String outputFile;
    private String dataset;
    private int nTrainingRelease;
    private float pctDataOnTraining;
    private float pctDefectiveInTraining;
    private float pctDefectiveInTesting;
    private String classifier;
    private String balancing;
    private String featureSelection;
    private String costSensitivity;

    public CSVRow(String outputFile, String dataset, int nTrainingRelease, float pctDataOnTraining, float pctDefectiveInTraining,
                  float pctDefectiveInTesting, String classifier, String balancing, String featureSelection, String costSensitivity) {
        this.outputFile = outputFile;
        this.dataset = dataset;
        this.nTrainingRelease = nTrainingRelease;
        this.pctDataOnTraining = pctDataOnTraining;
        this.pctDefectiveInTraining = pctDefectiveInTraining;
        this.pctDefectiveInTesting = pctDefectiveInTesting;
        this.classifier = classifier;
        this.balancing = balancing;
        this.featureSelection = featureSelection;
        this.costSensitivity = costSensitivity;
    }

    public String getOutputFile() {
        return outputFile;
    }

    public void setOutputFile(String outputFile) {
        this.outputFile = outputFile;
    }

    public String getDataset() {
        return dataset;
    }

    public void setDataset(String dataset) {
        this.dataset = dataset;
    }

    public int getnTrainingRelease() {
        return nTrainingRelease;
    }

    public void setnTrainingRelease(int nTrainingRelease) {
        this.nTrainingRelease = nTrainingRelease;
    }

    public float getPctDataOnTraining() {
        return pctDataOnTraining;
    }

    public void setPctDataOnTraining(float pctDataOnTraining) {
        this.pctDataOnTraining = pctDataOnTraining;
    }

    public float getPctDefectiveInTraining() {
        return pctDefectiveInTraining;
    }

    public void setPctDefectiveInTraining(float pctDefectiveInTraining) {
        this.pctDefectiveInTraining = pctDefectiveInTraining;
    }

    public float getPctDefectiveInTesting() {
        return pctDefectiveInTesting;
    }

    public void setPctDefectiveInTesting(float pctDefectiveInTesting) {
        this.pctDefectiveInTesting = pctDefectiveInTesting;
    }

    public String getClassifier() {
        return classifier;
    }

    public void setClassifier(String classifier) {
        this.classifier = classifier;
    }

    public String getBalancing() {
        return balancing;
    }

    public void setBalancing(String balancing) {
        this.balancing = balancing;
    }

    public String getFeatureSelection() {
        return featureSelection;
    }

    public void setFeatureSelection(String featureSelection) {
        this.featureSelection = featureSelection;
    }

    public String getCostSensitivity() {
        return costSensitivity;
    }

    public void setCostSensitivity(String costSensitivity) {
        this.costSensitivity = costSensitivity;
    }
}
