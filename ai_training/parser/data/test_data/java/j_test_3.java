public class TestJavaFile {

    public static int addNumbers(int a, int b) {
        return a + b;
    }

    public static void displayResult(int sum) {
        System.out.println("The sum is: " + sum);
    }

    public static void main(String[] args) {
        int result = addNumbers(10, 15);
        displayResult(result);
    }
}
