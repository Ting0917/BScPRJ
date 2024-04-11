public class TestJavaFile {
  public static void main(String[] args) {

      String firstPart = "Hello, ";
      String secondPart = "World!";
      String combinedString = firstPart + secondPart;
      System.out.println("Concatenated String: " + combinedString);
      
      String sub = combinedString.substring(7, 12);
      System.out.println("Extracted Substring: " + sub);
      
      int length = combinedString.length();
      System.out.println("Length of String: " + length);
  }
}

