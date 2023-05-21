using OpenCVSharp;
using OpenCvSharp.Extensions;
using System;


class Program

    static void Main()
    {
        using (var window = new Window("Reconocimiento Facial"))
        using (var capture = new VideoCapture(0))
        using (var faceCascade = new CascadeClassifier(@"haarcascade_frontalface_default.xml"))
        {
            if (!capture.IsOpened())
            {
                Console.WriteLine("No se puede abrir la cámara.");
                return;
            }

            while (true)
            {
                Mat frame = new Mat();
                capture.Read(frame);

                if (frame.Empty())
                    break;

                var grayFrame = new Mat();
                Cv2.CvtColor(frame, grayFrame, ColorConversionCodes.BGR2GRAY);
                Cv2.EqualizeHist(grayFrame, grayFrame);

                var faces = faceCascade.DetectMultiScale(grayFrame);

                foreach (var face in faces)
                {
                    Cv2.Rectangle(frame, face, Scalar.Red, 2);
                }

                window.ShowImage(frame);
                Cv2.WaitKey(1);
            }
        }
    }
}
