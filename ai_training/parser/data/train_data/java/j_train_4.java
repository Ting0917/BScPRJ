import java.util.Scanner;

public class TestJavaFile{

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.println("Enter an integer:");
        int myInt = scanner.nextInt();
        
        scanner.nextLine();
        System.out.println("Enter a string:");
        String myString = scanner.nextLine();
        
        System.out.println("You entered integer: " + myInt);
        System.out.println("You entered string: " + myString);
        
        scanner.close();
    }
}
