public class TestJavaFile {

    public static void main(String[] args) {
        System.out.println("For loop (1 to 5):");
        for(int i = 1; i <= 5; i++) {
            System.out.println(i);
        }

        System.out.println("While loop (5 to 1):");
        int count = 5;
        while(count > 0) {
            System.out.println(count);
            count--;
        }
    }
}
