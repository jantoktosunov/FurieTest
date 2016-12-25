#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include <string>
#include <iostream>
using namespace std;
using namespace cv;
//----------------------------------------------------------
// Функция для перестановки четвертей изображения местамии
// так, чтобы ноль спектра находился в центре изображения.
//----------------------------------------------------------
void Recomb(Mat &src,Mat &dst)
{
    int cx = src.cols>>1;
    int cy = src.rows>>1;
    Mat tmp;
    tmp.create(src.size(),src.type());
    src(Rect(0, 0, cx, cy)).copyTo(tmp(Rect(cx, cy, cx, cy)));
    src(Rect(cx, cy, cx, cy)).copyTo(tmp(Rect(0, 0, cx, cy)));
    src(Rect(cx, 0, cx, cy)).copyTo(tmp(Rect(0, cy, cx, cy)));
    src(Rect(0, cy, cx, cy)).copyTo(tmp(Rect(cx, 0, cx, cy)));
    dst=tmp;
}
//----------------------------------------------------------
// По заданному изображению рассчитывает
// действительную и мнимую части спектра Фурье
//----------------------------------------------------------
void ForwardFFT(Mat &Src, Mat *FImg)
{
    int M = getOptimalDFTSize( Src.rows );
    int N = getOptimalDFTSize( Src.cols );
    Mat padded;
    copyMakeBorder(Src, padded, 0, M - Src.rows, 0, N - Src.cols, BORDER_CONSTANT, Scalar::all(0));
    // Создаем комплексное представление изображения
    // planes[0] содержит само изображение, planes[1] его мнимую часть (заполнено нулями)
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg);
    dft(complexImg, complexImg);
    // После преобразования результат так-же состоит из действительной и мнимой части
    split(complexImg, planes);

    // обрежем спектр, если у него нечетное количество строк или столбцов
    planes[0] = planes[0](Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));
    planes[1] = planes[1](Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2));

    Recomb(planes[0],planes[0]);
    Recomb(planes[1],planes[1]);
    // Нормализуем спектр
    planes[0]/=float(M*N);
    planes[1]/=float(M*N);
    FImg[0]=planes[0].clone();
    FImg[1]=planes[1].clone();
}
//----------------------------------------------------------
// По заданным действительной и мнимой части
// спектра Фурье восстанавливает изображение
//----------------------------------------------------------
void InverseFFT(Mat *FImg,Mat &Dst)
{
    Recomb(FImg[0],FImg[0]);
    Recomb(FImg[1],FImg[1]);
    Mat complexImg;
    merge(FImg, 2, complexImg);
    // Производим обратное преобразование Фурье
    idft(complexImg, complexImg);
    split(complexImg, FImg);
    normalize(FImg[0], Dst, 0, 1, CV_MINMAX);
}
//----------------------------------------------------------
// Раскладывает изображение на амплитуду и фазу спектра Фурье
//----------------------------------------------------------
void ForwardFFT_Mag_Phase(Mat &src, Mat &Mag,Mat &Phase)
{
    Mat planes[2];
    ForwardFFT(src,planes);
    Mag.zeros(planes[0].rows,planes[0].cols,CV_32F);
    Phase.zeros(planes[0].rows,planes[0].cols,CV_32F);
    cv::cartToPolar(planes[0],planes[1],Mag,Phase);
}
//----------------------------------------------------------
// По заданным амплитуде и фазе
// спектра Фурье восстанавливает изображение
//----------------------------------------------------------
void InverseFFT_Mag_Phase(Mat &Mag,Mat &Phase, Mat &dst)
{
    Mat planes[2];
    planes[0].create(Mag.rows,Mag.cols,CV_32F);
    planes[1].create(Mag.rows,Mag.cols,CV_32F);
    cv::polarToCart(Mag,Phase,planes[0],planes[1]);
    InverseFFT(planes,dst);
}

int main(int argc, char ** argv)
{
    // Матрица изображения
    Mat img;
    // Ампльтуда спектра
    Mat Mag;
    // Фаза спектра
    Mat Phase;
    // Грузим изображение
    img=imread("/home/tokjan/ClionProjects/FurieTest/lena.jpg",0);
    imshow("Оригинал",img);

    // Раскладываем изображение в спектр
    ForwardFFT_Mag_Phase(img,Mag,Phase);

    //----------------------------------------------------------
    // Частотный фильтр
    //----------------------------------------------------------
    
    int R=300;  // Внешний радиус кольца пропускания
    int r=30;	// Внутренний радиус кольца пропускания
    Mat mask;
    mask.create(Mag.cols,Mag.rows,CV_32F);
    int cx = Mag.cols>>1;
    int cy = Mag.rows>>1;
    mask=1;
    cv::circle(mask,cv::Point(cx,cy),R,CV_RGB(0,0,0),-1);
    cv::circle(mask,cv::Point(cx,cy),r,CV_RGB(1,1,1),-1);
    //mask=1-mask; // Раскомментировать для инверсии
    cv::multiply(Mag,mask,Mag);
    cv::multiply(Phase,mask,Phase);

    InverseFFT_Mag_Phase(Mag,Phase,img);// Обратное преобразование

    //----------------------------------------------------------
    // Вывод результатов
    //----------------------------------------------------------
    // Преобразуем к виду удобному для отображения в окне
    Mat LogMag;
    LogMag.zeros(Mag.rows,Mag.cols,CV_32F);
    LogMag=(Mag+1);
    cv::log(LogMag,LogMag);


    imshow("Логарифм амплитуды", LogMag);
    imshow("Фаза", Phase);
    imshow("Результат фильтрации", img);

    cvWaitKey(0);
    return 0;
}
