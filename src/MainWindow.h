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

public slots:
	void fileNew();
	void fileOpen();
	void fileSave();
	void fileSaveAs();
	void aboutDialogShow();
	void setColorImage(const QImage& image);
	void setDepthImage(const QImage& image);

private:

	Ui::MainWindow *ui;
	QString currentFileName;
	QKinectPlayerCtrl* controller;
};

#endif // _MAIN_WINDOW_H_