public class TestJavaFile {

    static class Student {
        String name;
        int[] grades;

        Student(String name, int[] grades) {
            this.name = name;
            this.grades = grades;
        }
    }

    public static void main(String[] args) {
        Student student1 = new Student("Selina", new int[]{85, 92, 78, 90, 88});
        int sum = 0;
        for (int grade : student1.grades) {
            sum += grade;
        }
        double average = sum / 5.0;
        System.out.println("Average grade for " + student1.name + " is: " + average);
    }
}
