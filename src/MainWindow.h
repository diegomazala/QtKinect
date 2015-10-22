#ifndef _MAIN_WINDOW_H_
#define _MAIN_WINDOW_H_


#include <QMainWindow>

class QKinectPlayerCtrl;

namespace Ui 
{
	class MainWindow;
}


class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	
	explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

	void setController(QKinectPlayerCtrl* ctrl);
	
	//void setColorBuffer(const std::vector<uchar>& colorBuffer, ushort width, ushort height, ushort channels);
	//void setDepthBuffer(const std::vector<ushort>& depthBuffer, ushort width, ushort height);

public slots:
	void fileNew();
	void fileOpen();
	void fileSave();
	void fileSaveAs();
	void playerPlay();
	void playerStop();
	void playerRecord(bool triggered);
	void aboutDialogShow();

	void setColorImage(const QImage& image);
	void setDepthImage(const QImage& image);
	
signals:
	void recordToggled(bool);
	void play();
	void stop();

private:

	Ui::MainWindow *ui;
	QString currentFileName;
	QKinectPlayerCtrl* controller;
};

#endif // _MAIN_WINDOW_H_