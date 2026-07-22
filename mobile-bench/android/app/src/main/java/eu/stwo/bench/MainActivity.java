package eu.stwo.bench;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public final class MainActivity extends Activity {
    private static final String ACTION_TEST_LOOP = "com.google.intent.action.TEST_LOOP";
    private static final String EXTRA_SCENARIO = "scenario";
    private static final String OUTPUT_FILE_NAME = "bench.jsonl";
    private static final String TAG = "StwoBench";
    private static final int COPY_BUFFER_BYTES = 8192;
    private static final int DEFAULT_SCENARIO = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Intent launchIntent = getIntent();
        if (!ACTION_TEST_LOOP.equals(launchIntent.getAction())) {
            finish();
            return;
        }

        Uri outputUri = launchIntent.getData();
        if (outputUri == null) {
            throw new IllegalArgumentException("Game Loop launch did not provide an output URI");
        }

        int scenario = launchIntent.getIntExtra(EXTRA_SCENARIO, DEFAULT_SCENARIO);
        Log.i(TAG, "Starting scenario " + scenario + ", output=" + outputUri.getEncodedPath());

        new Thread(() -> runGameLoop(outputUri), "stwo-mobile-bench").start();
    }

    private void runGameLoop(Uri outputUri) {
        File cacheOutput = new File(getCacheDir(), OUTPUT_FILE_NAME);
        try {
            int lineCount = BenchRunner.runSuite(cacheOutput.getAbsolutePath());
            if (lineCount <= 0) {
                throw new IOException("Rust benchmark produced no JSON lines");
            }
            copyToTestLab(cacheOutput, outputUri);
            Log.i(TAG, "Finished benchmark with " + lineCount + " JSON lines");
            runOnUiThread(this::finish);
        } catch (Exception error) {
            Log.e(TAG, "Benchmark failed", error);
            throw new IllegalStateException("Game Loop benchmark failed", error);
        } finally {
            if (cacheOutput.exists() && !cacheOutput.delete()) {
                Log.w(TAG, "Could not delete cache output " + cacheOutput);
            }
        }
    }

    private void copyToTestLab(File source, Uri destination) throws IOException {
        try (InputStream input = new FileInputStream(source);
                OutputStream output = getContentResolver().openOutputStream(destination, "w")) {
            if (output == null) {
                throw new IOException("ContentResolver returned no output stream for " + destination);
            }
            copy(input, output);
        }
    }

    static void copy(InputStream input, OutputStream output) throws IOException {
        byte[] buffer = new byte[COPY_BUFFER_BYTES];
        int bytesRead;
        while ((bytesRead = input.read(buffer)) != -1) {
            output.write(buffer, 0, bytesRead);
        }
    }
}
