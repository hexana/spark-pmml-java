package main.java.com.exm.spark.funplay;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.model.MetroJAXBUtil;
import org.jpmml.sparkml.PMMLBuilder;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.Serializable;

public class PmmlCreator implements Serializable {
    private static final long serialVersionUID = 1L;

    public static void getPmmlFile(PipelineModel model, StructType schema,String outputPath){
        PMML pmml = new PMMLBuilder(schema, model).build();
        File file = new File(outputPath+"\\DT.pmml");
        try{
            OutputStream os = new FileOutputStream(file);
            MetroJAXBUtil.marshalPMML(pmml, os);
        }catch(Exception e){
            e.printStackTrace();
        }

    }
}
