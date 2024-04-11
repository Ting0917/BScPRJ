import java.io.IOException;

public class TestJavaFile {

    public static void main(String[] args) {
        System.out.println("Hello, World!");
        System.out.println("Press Enter to continue...");

        try {
            System.in.read();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
