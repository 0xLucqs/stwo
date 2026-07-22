package eu.stwo.bench;

final class BenchRunner {
    static {
        System.loadLibrary("bench_runner");
    }

    private BenchRunner() {}

    static native int runSuite(String outputPath);
}
