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

    public void createModel() {
        TFloat32 x = TFloat32.tensorOf(NdArrays.ofFloats(Shape.of(1, 6, 2)));
        //TFloat32 x = TFloat32.tensorOf(NdArrays.ofFloats(Shape.of(1, 2)));
        x.setFloat((float) 170, 0, 0, 0);
        x.setFloat((float) 60, 0, 0, 1);
        x.setFloat((float) 165, 0, 1, 0);
        x.setFloat((float) 55, 0, 1, 1);
        x.setFloat((float) 158, 0, 2, 0);
        x.setFloat((float) 50, 0, 2, 1);
        x.setFloat((float) 130, 0, 3, 0);
        x.setFloat((float) 25, 0, 3, 1);
        x.setFloat((float) 120, 0, 4, 0);
        x.setFloat((float) 20, 0, 4, 1);
        x.setFloat((float) 110, 0, 5, 0);
        x.setFloat((float) 18, 0, 5, 1);

        TFloat32 y = TFloat32.tensorOf(NdArrays.ofFloats(Shape.of(1, 6)));
        y.setFloat(1, 0, 0);
        y.setFloat(1, 0, 1);
        y.setFloat(1, 0, 2);
        y.setFloat(0, 0, 3);
        y.setFloat(0, 0, 4);
        y.setFloat(0, 0, 5);

    }
// x = np.array([[170, 60], [165, 55], [158, 50], [130, 25], [120, 20], [110, 18]])
// y = np.array([[1], [1], [1], [0], [0], [0]])

// model = tf.keras.Sequential()
// model.add(tf.keras.layers.Dense(1, input_dim=2, use_bias=True, activation="sigmoid"))
// model.compile(optimizer="adam", loss="binary_crossentropy", metrics="binary_accuracy")
// model.fit(x, y, epochs=1000)
// print(model.summary())

}
