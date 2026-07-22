package eu.stwo.bench;

import static org.junit.Assert.assertArrayEquals;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

import org.junit.Test;

public final class MainActivityTest {
    @Test
    public void copyPreservesMultipleBuffers() throws Exception {
        byte[] expected = new byte[16_401];
        for (int index = 0; index < expected.length; index++) {
            expected[index] = (byte) (index * 31);
        }

        ByteArrayOutputStream actual = new ByteArrayOutputStream();
        MainActivity.copy(new ByteArrayInputStream(expected), actual);

        assertArrayEquals(expected, actual.toByteArray());
    }
}
