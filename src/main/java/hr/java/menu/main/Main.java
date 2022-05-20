package hr.java.menu.main;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationSoftMax;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Main {
    /* karakteristike:
        - duljina rijeci
        - prvo slovo ascii
        - zadnje slovo ascii
        - postotak samoglasnika u rijeci (a,e,i,o,u) x/5
        - broj samoglasnika
        - zbroj ascii vrijednosti
        - 26 neurona svaki oznacava postotak slova abecede u cijeloj rijeci
    * */
    public static double[][] input;
    public static double[][] ideal_output;
    public static int neuroni_input = 32;
    public static int neuroni_hidden = 64;
    public static int neuroni_output = 15;

    public static List<Double> letters(String word){
        List<Character> slova = new ArrayList<>();
        List<Character> pom = new ArrayList<>();
        List<Double> rjesenja = new ArrayList<>();

        for(int i = 0;i < 26;i++){
            pom.add((char) (97 + i));
        }
        for(int i = 0;i < word.length();i++){
            slova.add(word.charAt(i));
        }
        for(int i = 0;i < 26;i++){
            rjesenja.add(0.0);
        }
        for(int i = 0;i < pom.size();i++){
            for(int j = 0;j < word.length();j++){
                if(word.charAt(j) == pom.get(i)){
                   rjesenja.set(i,rjesenja.get(i) + 1);
                }
            }
        }
        double velicina = word.length();
        for(int i = 0;i < 26;i++){
            rjesenja.set(i,rjesenja.get(i) / velicina);
        }

        return rjesenja;

    }

    public static double asciiSum(String word){

        int a = 0;
        double sum = 0;
        for(int i = 0;i < word.length();i++){
            a = word.charAt(i) - 97;
            sum = sum + a;
        }
        return sum;
    }

    public static double vowelsCount(String word){

        double count = 0;
        for(int i = 0;i < word.length();i++){
            if( (word.charAt(i) == 'a') || (word.charAt(i) == 'e') || (word.charAt(i) == 'i') ||
                    (word.charAt(i) == 'o') || (word.charAt(i) == 'u')){
                count++;
            }
        }
        return count;
    }
    public static double vowelsPercentage(String word){

        double count = 0;
        List<Character> slova = new ArrayList<>();
        for(int i = 0;i < word.length();i++){
            slova.add(word.charAt(i));
        }

        if(slova.contains('a')){
            count++;
        }
        if(slova.contains('e')){
            count++;
        }
        if(slova.contains('i')){
            count++;
        }
        if(slova.contains('o')){
            count++;
        }
        if(slova.contains('u')){
            count++;
        }
        return count;
    }
    public static List<Double> normalize(int new_min, int new_max, double min, double max,List<Double> arr){
        List<Double> lista = new ArrayList<>();
        for(int i = 0 ; i < arr.size() ; i++){
            double v = ((arr.get(i) - min)/(max - min))*(new_max - new_min) + new_min;
            lista.add(v);
        }
        return lista;
    }
    public static double[][] inputData() throws IOException {
        String st;
        double min,max;
        List<Double> duljine = new ArrayList<>();
        List<Double> prvoSlovo = new ArrayList<>();
        List<Double> zadnjeSlovo = new ArrayList<>();
        List<Double> samoglasniciBroj = new ArrayList<>();
        List<Double> samoglasniciPostotak = new ArrayList<>();
        List<Double> asciiZbroj =new ArrayList<>();
        List<List<Double>> slova =new ArrayList<>();


        BufferedReader br = new BufferedReader(new FileReader("rijeci"));
        while((st = br.readLine()) != null){
           duljine.add((double) st.length());
           prvoSlovo.add((double) st.charAt(0));
           zadnjeSlovo.add((double) st.charAt(st.length() - 1));
           samoglasniciBroj.add(vowelsCount(st));
           samoglasniciPostotak.add(vowelsPercentage(st) / 5);
           asciiZbroj.add(asciiSum(st));
           slova.add(letters(st));

        }
        min = duljine.stream().mapToDouble(value -> value).min().getAsDouble();
        max = duljine.stream().mapToDouble(value -> value).max().getAsDouble();
        duljine = normalize(0,1,min,max,duljine);

        min = prvoSlovo.stream().mapToDouble(value -> value).min().getAsDouble();
        max = prvoSlovo.stream().mapToDouble(value -> value).max().getAsDouble();
        prvoSlovo = normalize(0,1,min,max,prvoSlovo);

        min = zadnjeSlovo.stream().mapToDouble(value -> value).min().getAsDouble();
        max = zadnjeSlovo.stream().mapToDouble(value -> value).max().getAsDouble();
        zadnjeSlovo = normalize(0,1,min,max,zadnjeSlovo);

        min = samoglasniciBroj.stream().mapToDouble(value -> value).min().getAsDouble();
        max = samoglasniciBroj.stream().mapToDouble(value -> value).max().getAsDouble();
        samoglasniciBroj = normalize(0,1,min,max,samoglasniciBroj);

        min = asciiZbroj.stream().mapToDouble(value -> value).min().getAsDouble();
        max = asciiZbroj.stream().mapToDouble(value -> value).max().getAsDouble();
        asciiZbroj = normalize(0,1,min,max,asciiZbroj);


        // duljine.size() - broj rijeci
        // n - broj karakteristika rijeci
        int n = 32;
        double[][] karakteristike = new double[duljine.size()][n];
        for(int i = 0;i < duljine.size();i++){
            karakteristike[i][0] = duljine.get(i);
            karakteristike[i][1] = prvoSlovo.get(i);
            karakteristike[i][2] = zadnjeSlovo.get(i);
            karakteristike[i][3] = samoglasniciBroj.get(i);
            karakteristike[i][4] = samoglasniciPostotak.get(i);
            karakteristike[i][5] = asciiZbroj.get(i);
            for(int j = 0;j < 26;j++){
                karakteristike[i][6 + j] = slova.get(i).get(j);
            }
        }

        return karakteristike;
    }
    public static double[][] idealOutputData(double[][] inputD) throws IOException {
        String st;
        BufferedReader br = new BufferedReader(new FileReader("prijevodi"));
        // inputD.length - broj elemenata u data setu,broj rijeci
        // one hot encoding
        int n = 15;
        double[][] outputValues = new double[inputD.length][n];

        for(int i = 0;i < outputValues.length;i++){
            for(int j = 0;j < n;j++){
                outputValues[i][j] = 0;
            }
        }
        int a = 1;
        int b = 0;
        for(int i = 0;i < inputD.length;i++){
            if(a > 6){
                a = 1;
                b++;
            }
            outputValues[i][b] = 1;
            a++;
        }

        return outputValues;

    }
    public static void main(String[] args) throws IOException {

        input = inputData();
        ideal_output = idealOutputData(input);

        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null,true,neuroni_input));
        network.addLayer(new BasicLayer(new ActivationSigmoid(),true,neuroni_hidden));
        network.addLayer(new BasicLayer(new ActivationSigmoid(),true,neuroni_hidden));
        network.addLayer(new BasicLayer(new ActivationSigmoid(),false,neuroni_output));
        network.getStructure().finalizeStructure();
        network.reset();

        MLDataSet mlDataSet = new BasicMLDataSet(input,ideal_output);
        final MLTrain train = new QuickPropagation(network,mlDataSet);
        int i = 1;
        int epohe = 0;
        do{
            train.iteration();
            System.out.println("Error: " + train.getError() + " Epoch: " + i);
            i++;

        }while (train.getError() > 0.001);
        epohe = i;

        List<Double> op = new ArrayList<>();
        List<Double> id = new ArrayList<>();
        double max_op = 0;
        double max_id = 0;
        int index_op = 0;
        int index_id = 0;
        int tocno = 0;
        for(MLDataPair pair : mlDataSet){
            final MLData output = network.compute(pair.getInput());
            System.out.println("input=" + pair.getInput());
            System.out.println("output=" + output);
            System.out.println("ideal=" + pair.getIdeal());
            System.out.println();

            op.clear();
            id.clear();
            for(int x = 0;x < output.size();x++){
                op.add(output.getData(x));
            }
            for(int x = 0;x < pair.getIdeal().size();x++){
                id.add(pair.getIdeal().getData(x));
            }
            max_op = op.stream().mapToDouble(value -> value).max().getAsDouble();
            max_id = id.stream().mapToDouble(value -> value).max().getAsDouble();
            for(int x = 0;x < op.size();x++){
                if(op.get(x) == max_op){
                    index_op = x;
                }
            }
            for(int x = 0;x < id.size();x++){
                if(id.get(x) == max_id){
                    index_id = x;
                }
            }

            if(index_op == index_id){
                tocno++;
            }
        }
        double rjesenje = tocno / 90.0;
        System.out.println("Rjesenost: " + String.format("%.2f",rjesenje * 100.0) + "%");
        System.out.println(epohe + "tt" + tocno);

    }
}
