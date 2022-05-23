import java.io.File;
import java.net.URI;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.types.TFloat32;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;


public class TFTestClient {

    private SavedModelBundle model;

    public boolean isLoaded() {
        return model != null;
    }

    public void load() {
        loadModel();
    }

    public void unload() {
        model.close();
    }

    /**
     * Load TF model.
     */
    private void loadModel() {
        try {
            URI modelURI = getClass().getResource("/my_model").toURI();
            System.out.println(modelURI);

            File file = new File(modelURI);
            String modelPath = file.getPath();
            System.out.println(modelPath);

            model = (SavedModelBundle) SavedModelBundle.load(modelPath);

            // def buffer = loadModelFile(Config.TFL_MODEL_FILE)
            // tflite = Interpreter(buffer)

            System.out.println("TF model loaded.");
            printTensor();

        } catch (Exception ex) {
            ex.printStackTrace();
        }

    }

    public Object predict(final Integer input1, final Integer input2) {

        TFloat32 input = TFloat32.tensorOf(NdArrays.ofFloats(Shape.of(1, 2)));
        input.setFloat((float) input1, 0, 0);
        input.setFloat((float) input2, 0, 1);

        TFloat32 output = (TFloat32) model.session()
            .runner()
            .feed("serving_default_dense_1_input:0", input)
            .fetch("StatefulPartitionedCall:0")
            .run()
            .get(0);

        System.out.println();
        System.out.println("predict(" + input1 + ", " + input2 + ")");
        System.out.println("prediction = " + output.getFloat(0, 0));
        System.out.println("size = " + output.size());
        System.out.println("shape = " + output.shape());
        return output.getFloat(0, 0);
    }

    public void testSavedModelBundle() {
        System.out.println("---------- Start testSavedModelBundle");
        load();
        predict(175, 80);
        predict(120, 22);
        unload();
        System.out.println("---------- End testSavedModelBundle");
    }

    private void printTensor() {
        System.out.println();        
        model.graph().operations().forEachRemaining(System.out::println);
        // print(model.graph().toGraphDef())
    }


}
