

#ifndef _Q_IMAGE_WIDGET_H
#define _Q_IMAGE_WIDGET_H

#include <QLabel>


class QImageWidget : public QLabel
{
    Q_OBJECT

public:
    QImageWidget(QWidget* parent = nullptr);
    

public slots:
	virtual bool loadFile(const QString &);
	virtual void setImage(const QImage& image);
	virtual void save(const QString& filename);


Q_SIGNALS:
	void closed();

protected:
	//virtual void paintEvent(QPaintEvent *);
	virtual void keyReleaseEvent(QKeyEvent *);

	virtual void closeEvent(QCloseEvent *event);
};

#endif	// _Q_IMAGE_WIDGET_H
