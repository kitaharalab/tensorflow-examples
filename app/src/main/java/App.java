public class App {

    public static void main(String[] args) {
        App app = new App();
        app.testTFL();
    }

    public void testTFL() {
        TFTestClient tfClient = new TFTestClient();
        tfClient.testSavedModelBundle();
    }

}
